package dev.vkazulkin;

public record AsyncEmbeddingResponse(Float[] embedding, String status, SegmentMetadata segmentMetadata) {

    public static record SegmentMetadata (int segmentIndex, int segmentStartSeconds, int segmentEndSeconds)  {}
}
