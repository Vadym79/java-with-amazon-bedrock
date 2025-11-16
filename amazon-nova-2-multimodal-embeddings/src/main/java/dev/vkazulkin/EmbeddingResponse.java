package dev.vkazulkin;

import java.util.List;

public record EmbeddingResponse (List<Embeddings> embeddings) {

	public static record Embeddings (String embeddingType, Float[] embedding) {
    }
}
