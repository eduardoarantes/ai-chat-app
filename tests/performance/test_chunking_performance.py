"""Performance tests for chunking functionality."""

import time

import pytest

from document_processing.processors import ChunkOptimizer, RecursiveCharacterChunkingStrategy, TextChunkingStrategy
from models.chunking import ChunkingConfig, ChunkingStrategy


class TestChunkingPerformance:
    """Test chunking performance and benchmarks."""

    @pytest.fixture
    def large_text(self):
        """Generate large text for performance testing."""
        # Create approximately 50KB of text
        paragraph = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

        return "\n\n".join([f"Paragraph {i}: {paragraph}" for i in range(200)])

    @pytest.fixture
    def standard_config(self):
        """Standard chunking configuration."""
        return ChunkingConfig(chunk_size=1000, chunk_overlap=200, min_chunk_size=100, max_chunk_size=2000)

    def test_recursive_character_performance(self, large_text, standard_config):
        """Test performance of recursive character chunking."""
        strategy = RecursiveCharacterChunkingStrategy(standard_config)

        start_time = time.time()
        chunks = strategy.chunk_text(large_text, {"document_id": "perf_test"})
        end_time = time.time()

        processing_time = end_time - start_time

        # Performance metrics (commented out to avoid console pollution)
        # print(f"\nRecursive Character Chunking Performance:")
        # print(f"Text length: {len(large_text):,} characters")
        # print(f"Chunks created: {len(chunks)}")
        # print(f"Processing time: {processing_time:.4f} seconds")
        # print(f"Characters per second: {len(large_text) / processing_time:,.0f}")

        # Performance assertions
        assert processing_time < 5.0  # Should process within 5 seconds
        assert len(chunks) > 0
        assert all(chunk.size >= standard_config.min_chunk_size for chunk in chunks)

    def test_text_strategy_performance(self, large_text, standard_config):
        """Test performance of text chunking strategy."""
        strategy = TextChunkingStrategy(standard_config)

        start_time = time.time()
        chunks = strategy.chunk_text(large_text, {"document_id": "perf_test"})
        end_time = time.time()

        processing_time = end_time - start_time

        # Performance metrics (commented out to avoid console pollution)
        # print(f"\nText Strategy Chunking Performance:")
        # print(f"Text length: {len(large_text):,} characters")
        # print(f"Chunks created: {len(chunks)}")
        # print(f"Processing time: {processing_time:.4f} seconds")
        # print(f"Characters per second: {len(large_text) / processing_time:,.0f}")

        # Performance assertions
        assert processing_time < 5.0  # Should process within 5 seconds
        assert len(chunks) > 0

    def test_chunk_optimizer_performance(self, large_text, standard_config):
        """Test performance of chunk size optimization."""
        optimizer = ChunkOptimizer(standard_config)

        start_time = time.time()
        optimized_size = optimizer.optimize_chunk_size(large_text, {"complexity_score": 0.5})
        token_count = optimizer.count_tokens(large_text)
        end_time = time.time()

        processing_time = end_time - start_time

        # Performance metrics (commented out to avoid console pollution)
        # print(f"\nChunk Optimizer Performance:")
        # print(f"Text length: {len(large_text):,} characters")
        # print(f"Token count: {token_count:,}")
        # print(f"Optimized chunk size: {optimized_size}")
        # print(f"Processing time: {processing_time:.4f} seconds")

        # Performance assertions
        assert processing_time < 2.0  # Optimization should be fast
        assert standard_config.min_chunk_size <= optimized_size <= standard_config.max_chunk_size
        assert token_count > 0

    def test_memory_usage_during_chunking(self, large_text, standard_config):
        """Test that memory usage remains reasonable during chunking."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        strategy = RecursiveCharacterChunkingStrategy(standard_config)
        chunks = strategy.chunk_text(large_text, {"document_id": "memory_test"})

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory usage metrics (commented out to avoid console pollution)
        # print(f"\nMemory Usage During Chunking:")
        # print(f"Initial memory: {initial_memory:.2f} MB")
        # print(f"Peak memory: {peak_memory:.2f} MB")
        # print(f"Memory increase: {memory_increase:.2f} MB")
        # print(f"Memory per chunk: {memory_increase / len(chunks):.4f} MB")

        # Memory assertions
        assert memory_increase < 100  # Should not use more than 100MB additional
        assert len(chunks) > 0

    def test_concurrent_chunking_performance(self, large_text, standard_config):
        """Test performance with multiple concurrent chunking operations."""
        import concurrent.futures
        import threading

        def chunk_text_task(text, task_id):
            strategy = RecursiveCharacterChunkingStrategy(standard_config)
            return strategy.chunk_text(text, {"document_id": f"concurrent_test_{task_id}"})

        start_time = time.time()

        # Run 5 concurrent chunking tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(chunk_text_task, large_text, i) for i in range(5)]

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        processing_time = end_time - start_time

        # Concurrent performance metrics (commented out to avoid console pollution)
        # print(f"\nConcurrent Chunking Performance:")
        # print(f"Number of tasks: 5")
        # print(f"Text length per task: {len(large_text):,} characters")
        # print(f"Total processing time: {processing_time:.4f} seconds")
        # print(f"Average chunks per task: {sum(len(chunks) for chunks in results) / len(results):.1f}")

        # Performance assertions
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 5
        assert all(len(chunks) > 0 for chunks in results)

    def test_chunking_quality_metrics(self, large_text, standard_config):
        """Test quality metrics of chunking output."""
        strategy = RecursiveCharacterChunkingStrategy(standard_config)
        chunks = strategy.chunk_text(large_text, {"document_id": "quality_test"})

        # Calculate quality metrics
        total_original_length = len(large_text)
        total_chunked_length = sum(len(chunk.content) for chunk in chunks)

        # Check overlap efficiency
        overlaps = []
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]

            # Simple overlap detection
            overlap = 0
            for j in range(min(len(chunk1.content), standard_config.chunk_overlap)):
                if chunk1.content[-j - 1 :] and chunk2.content[: j + 1]:
                    if chunk1.content[-j - 1 :] == chunk2.content[: j + 1]:
                        overlap = j + 1

            overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

        # Boundary quality assessment
        good_boundaries = 0
        for chunk in chunks:
            if chunk.content.rstrip().endswith((".", "!", "?", "\n\n")):
                good_boundaries += 1

        boundary_quality = good_boundaries / len(chunks) if chunks else 0

        # Quality metrics (commented out to avoid console pollution)
        # print(f"\nChunking Quality Metrics:")
        # print(f"Total chunks: {len(chunks)}")
        # print(f"Original text length: {total_original_length:,}")
        # print(f"Total chunked length: {total_chunked_length:,}")
        # print(f"Content preservation: {(total_chunked_length / total_original_length) * 100:.1f}%")
        # print(f"Average overlap: {avg_overlap:.1f} characters")
        # print(f"Boundary quality: {boundary_quality * 100:.1f}%")

        # Quality assertions
        assert len(chunks) > 10  # Should create reasonable number of chunks
        assert total_chunked_length >= total_original_length * 0.95  # Should preserve at least 95% of content
        assert boundary_quality > 0.5  # At least 50% should have good boundaries
        assert avg_overlap <= standard_config.chunk_overlap  # Overlap should be within limits


if __name__ == "__main__":
    # Run performance tests directly
    pytest.main([__file__, "-v", "-s"])
