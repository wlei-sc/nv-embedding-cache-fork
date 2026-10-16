/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <embedding_layer.hpp>
#include <vector>
#include "layer_utils.hpp"

namespace nve {
 
template <typename KeyType> class GpuTable;
class InsertHeuristic;

/**
 * An embedding layer with a hierarchy of tables (caches/databases).
 * Lookups will be resolved by processing tables in order, such that indices missing in one table will be forwarded to the next.
 */
template <typename KeyType>
class HierarchicalEmbeddingLayer : public EmbeddingLayerBase {
 public:
  struct Config {
    std::string layer_name;
    std::shared_ptr<InsertHeuristic> insert_heuristic = nullptr; // nullptr will result in using the default InsertHeuristic
                                                                 // auto inserts can be disabled by using the NeverInsertHeuristic class
    int64_t min_insert_freq_gpu = 0; // Minimal amount of lookups between inserts on GPU. increase this to throttle down auto inserts
    int64_t min_insert_freq_host = 0; // Minimal amount of lookups between inserts on Host. increase this to throttle down auto inserts
    int64_t min_insert_size_gpu = 1 << 16; // Minimal amount of keys to trigger an insert on GPU (smaller amounts will be collected)
    int64_t min_insert_size_host = 0; // Minimal amount of keys to trigger an insert on Host (smaller amounts will be collected)
    std::vector<uint8_t> default_embedding = {}; // Default embedding vector bytes returned for keys missing from all tables. Empty implies no default (misses undefined).
  };

  NVE_PREVENT_COPY_AND_MOVE_(HierarchicalEmbeddingLayer);
  using key_type = KeyType;

  /**
   * Create a Hierarchical embedding layer, using multiple tables.
   * 
   * @param cfg Layer configuration parameters.
   * @param tables Vector of valid table shared_ptrs to use.
   *               Tables will be handled in order during lookup.
   *               i.e. table performance is expected to decrease along the table vector
   * @param allocator Allocator to use for internal resources not bound to a specific context
   *                  nullptr implies using the default allocator
   */
  HierarchicalEmbeddingLayer(const Config& cfg, const std::vector<table_ptr_t>& tables,
                             allocator_ptr_t allocator = {});

  ~HierarchicalEmbeddingLayer() override;

  void lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
              const int64_t output_stride, bitmask64_t* hitmask,
              const PoolingParams* pool_params, float* hitrates) override;
  void insert(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size, const void* values,
              int64_t table_id) override;
  void update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size,
              const void* values, const int64_t table_id) override;
  void accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                  const int64_t value_stride, const int64_t value_size, const void* values,
                  DataType_t value_type, const int64_t table_id) override;
  void clear(context_ptr_t& ctx) override;
  void erase(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
             int64_t table_id) override;
  context_ptr_t create_execution_context(
    cudaStream_t lookup_stream, cudaStream_t modify_stream, thread_pool_ptr_t thread_pool, allocator_ptr_t allocator) override;
  int64_t get_num_tables() const override;

  /**
   * Keeps keys missing from every table in the cache tables (all tables but the last).
   * Auto-insert then stores such a key under its own key, with the default embedding as its row,
   * instead of dropping it, so later lookups resolve it from the caches like any other key rather
   * than asking the last table again. The last table, the backing store the misses came from,
   * never receives them. Off by default. Requires a `default_embedding`. Call before the first
   * lookup.
   */
  void set_cache_missed_keys(bool enable);

  /**
   * An asynchronous lookup in three calls on one context, one lookup in flight per context.
   * Same results and same cache fill as lookup(); only where the host work runs changes:
   * - lookup_async_begin: on the lookup stream, the GPU table's find and the copy of the keys and
   *   GPU hit mask to the host. A thread-pool task then runs the other tables' finds and the
   *   default rows on host memory; besides waiting for that copy it makes no CUDA calls.
   * - lookup_async_complete: waits for the task, then queues on the lookup stream the scatter of
   *   the rows the GPU table missed into `output`, and a copy of `output` for the cache fill.
   * - lookup_async_finish: once the lookup stream has passed complete, runs the cache fill from
   *   that copy and reports per-table hit rates (as lookup()'s `hitrates`) and stats. Must follow
   *   every begin, also when complete was skipped or threw: it then only waits for the task.
   * Requires supports_async_lookup(). No pooling, no output hit mask.
   *
   * With `defer_last_table`, the task skips the last table (the backing store), so a caller can
   * fetch for several layers at once: lookup_async_deferred waits for the task and hands over that
   * table's find on the task's host buffers, whose unresolved rows already hold the default row;
   * the caller writes the rows it finds and sets their hit bits, then passes how many it found to
   * lookup_async_complete.
   */
  bool supports_async_lookup() const;
  void lookup_async_begin(context_ptr_t& ctx, int64_t num_keys, const void* keys, void* output,
                          int64_t output_stride, bool defer_last_table = false);
  AsyncDeferredFind lookup_async_deferred(context_ptr_t& ctx);
  void lookup_async_complete(context_ptr_t& ctx, int64_t deferred_hits = 0);
  void lookup_async_finish(context_ptr_t& ctx, float* hitrates, AsyncLookupStats* stats);

  inline const Config& get_config() const { return config_; }

 private:
  // The async lookup's thread-pool task: the host tables' and backing store's finds (the backing
  // store's unless deferred), then the default rows, all on the host copies.
  void run_async_host_stage(HierarchicalAsyncLookup& lookup) const;

  const Config config_;
  allocator_ptr_t allocator_;
  std::vector<table_ptr_t> tables_;
  int32_t gpu_device_; // gpu device id or -1

  // Either empty (auto-insert disabled) or one handler per table.
  std::vector<std::shared_ptr<AutoInsertHandler>> auto_insert_handlers_;

  // Handler for table `i`, or null when auto-insert is disabled. Lets modify ops take a
  // ScopedModifyLock unconditionally.
  inline const std::shared_ptr<AutoInsertHandler>& handler_at(size_t i) const {
    static const std::shared_ptr<AutoInsertHandler> none;
    return auto_insert_handlers_.empty() ? none : auto_insert_handlers_.at(i);
  }
};

}  // namespace nve
