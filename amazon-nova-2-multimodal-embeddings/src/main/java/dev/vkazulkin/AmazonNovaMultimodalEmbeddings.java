package dev.vkazulkin;

import com.fasterxml.jackson.databind.ObjectMapper;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.core.document.Document;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.bedrockruntime.BedrockRuntimeClient;
import software.amazon.awssdk.services.bedrockruntime.model.*;
import software.amazon.awssdk.services.s3vectors.S3VectorsClient;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3vectors.model.*;

public class AmazonNovaMultimodalEmbeddings {

	private static final String MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0";
	private static final int EMBEDDING_DIMENSION = 384;

	private static final String AWS_LAMBDA_EMBEDDINGS = "AWS Lambda is a serverless compute service for running code without having to provision or manage servers. You pay only for the compute time you consume.";
	private static final String AZURE_FUNCTIONS__EMBEDDINGS = "Azure Functions is a serverless solution that allows you to build robust apps while using less code, and with less infrastructure and lower costs.";

	private final static String S3_BUCKET = "s3://vk-amazon-nova-2-mme/";

	private final static String S3_EMBEDDINGS_DESTINATION_URI = S3_BUCKET + "embeddings-output/";

	private final static String IMAGE_EXTENSION = ".jpg";
	private static final String[] IMAGE_NAMES = { "AWS-Lambda", "Azure-Functions" };

	private final static String AUDIO_EXTENSION = ".mp3";
	private static final String[] AUDIO_NAMES = { "AWS-Lambda-explained-in-90-seconds-audio" };

	private final static String VIDEO_EXTENSION = ".mp4";
	private static final String[] VIDEO_NAMES = { "AWS-Lambda-explained-in-90-seconds-video" };

	private static final String VECTOR_BUCKET = "vk-vector-store";
	private static final String INDEX_NAME = "embeddings";

	private static final BedrockRuntimeClient BEDRDOCK_RUNTIME_CLIENT = BedrockRuntimeClient.builder()
			.credentialsProvider(DefaultCredentialsProvider.builder().build()).region(Region.US_EAST_1).build();

	private static final S3VectorsClient S3_VECTORS_CLIENT = S3VectorsClient.builder()
			.credentialsProvider(DefaultCredentialsProvider.builder().build()).region(Region.US_EAST_1).build();

	private static final S3Client S3_CLIENT = S3Client.builder()
			.credentialsProvider(DefaultCredentialsProvider.builder().build()).region(Region.US_EAST_1).build();

	private static final ObjectMapper MAPPER = new ObjectMapper();

	/**
	 * creates S3 Vector bucket and index
	 */
	private static void createS3VectorBucketAndIndex() {

		var cvbRequest = CreateVectorBucketRequest.builder().vectorBucketName(VECTOR_BUCKET).build();
		S3_VECTORS_CLIENT.createVectorBucket(cvbRequest);

		var ciRequest = CreateIndexRequest.builder().vectorBucketName(VECTOR_BUCKET).indexName(INDEX_NAME)
				.dataType("float32").dimension(EMBEDDING_DIMENSION).distanceMetric("cosine").build();

		S3_VECTORS_CLIENT.createIndex(ciRequest);
	}

	/**
	 * searches for text semantically in the S3 Vectors
	 * 
	 * @param text - text to search
	 * @param topK - number of closest results to search
	 * @throws Exception
	 */
	private static void search(String text, int topK) throws Exception {
		Float[] embeddings = createTextEmbeddings(text, "GENERIC_RETRIEVAL");
		var vd = VectorData.builder().float32(embeddings).build();
		var qvrRequest = QueryVectorsRequest.builder().vectorBucketName(VECTOR_BUCKET).indexName(INDEX_NAME).topK(topK)
				.returnDistance(true).returnMetadata(true).queryVector(vd).build();
		var qvResponse = S3_VECTORS_CLIENT.queryVectors(qvrRequest);
		for (var qov : qvResponse.vectors()) {
			System.out.println("vector: " + qov);
		}

	}

	/**
	 * creates text embeddings with the given text and returns it as float array
	 * 
	 * @param text - text
	 * @return text embeddings as float array
	 * @throws Exception
	 */
	private static Float[] createTextEmbeddings(String text, String embeddingPurpose) throws Exception {
		String request = """
				{
				    "taskType": "SINGLE_EMBEDDING",
				    "singleEmbeddingParams": {
				        "embeddingPurpose": {{embeddingPurpose}},
				        "embeddingDimension": {{dimension}},
				        "text": {"truncationMode": "END", "value": {{text}} }
				      }
				}""".replace("{{text}}", "\"" + text + "\"")
				.replace("{{dimension}}", String.valueOf(EMBEDDING_DIMENSION))
				.replace("{{embeddingPurpose}}", "\"" + embeddingPurpose + "\"");

		System.out.println(request);
		var eResponse = invokeBedrockModel(request);
		System.out.println("embedding:" + eResponse.embeddings().getFirst().embedding());
		return eResponse.embeddings().getFirst().embedding();
	}

	/**
	 * Creates and stores text embeddings with the given text and key into the S3
	 * Vector.
	 * 
	 * @param text - text
	 * @param key  - key
	 * @throws Exception
	 */
	private static void createAndStoreTextEmbeddings(String text, String key) throws Exception {
		Float[] embeddings = createTextEmbeddings(text, "GENERIC_INDEX");
		putVectors(embeddings, key);
	}

	/**
	 * Creates and stores image file embeddings into the S3 Vector. Image file name
	 * without extension will be used as a key
	 * 
	 * @throws Exception
	 */
	private static void createAndStoreImageEmbeddings() throws Exception {
		for (String imageName : IMAGE_NAMES) {
			String request = """
					{
					    "taskType": "SINGLE_EMBEDDING",
					    "segmentedEmbeddingParams": {
					        "embeddingPurpose": "GENERIC_INDEX",
					        "embeddingDimension": {{dimension}},
					        "image": {
					            "format": "jpeg",
					            "source": {
					                 "s3Location": {"uri": {{S3_IMAGE_URI}} }
					             }
					          }
					      }
					}""".replace("{{S3_IMAGE_URI}}", "\"" + S3_BUCKET + imageName + IMAGE_EXTENSION + "\"")
					.replace("{{dimension}}", String.valueOf(EMBEDDING_DIMENSION));

			System.out.println(request);
			var eResponse = invokeBedrockModel(request);
			System.out.println("embedding: " + eResponse.embeddings().getFirst().embedding());

			putVectors(eResponse.embeddings().getFirst().embedding(), imageName);
		}
	}

	/**
	 * Creates and stores audio file embeddings into the S3 Vector. Audio file name
	 * without extension will be used as a key
	 * 
	 * @throws Exception
	 */
	private static void createAndStoreAudioEmbeddings() throws Exception {
		for (String audioName : AUDIO_NAMES) {
			asyncInvokeBerockModelAndPutVectorsToS3(prepareAudioDocument(S3_BUCKET + audioName + AUDIO_EXTENSION), audioName,
				"embedding-audio.jsonl");
		}
	}

	/**
	 * Creates and stores video file embeddings into the S3 Vector. Video file name
	 * without extension will be used as a key
	 * 
	 * @throws Exception
	 */
	private static void createAndStoreVideoEmbeddings() throws Exception {
		for (String videoName : VIDEO_NAMES) {
			asyncInvokeBerockModelAndPutVectorsToS3(prepareVideoDocument(S3_BUCKET + videoName + VIDEO_EXTENSION), videoName,
				"embedding-audio-video.jsonl");
		}
	}

	/**
	 * invokes Amazon Bedrock Model with the given text and returns its response's
	 * body as EmbeddingResponse object
	 * 
	 * @param request - request to execute
	 * @return response's body as EmbeddingResponse object
	 * @throws Exception
	 */
	private static EmbeddingResponse invokeBedrockModel(String request) throws Exception {
		var imRequest = InvokeModelRequest.builder().modelId(MODEL_ID).body(SdkBytes.fromUtf8String(request))
				.contentType("application/json").accept("application/json").build();
		var imResponse = BEDRDOCK_RUNTIME_CLIENT.invokeModel(imRequest);
		System.out.println(imResponse.body().asUtf8String());
		return MAPPER.readValue(imResponse.body().asUtf8String(), EmbeddingResponse.class);
	}

	
	/** invokes the Amazon Bedrock Model asynchronously and stores the embeddings in the s3 vectors
	 * 
	 * @param document document to be used as the model input
	 * @param fileName - file name which is used as a key prefix in the s3 vectors
	 * @param embeddingsResultFileName - file name with the embedding results computed asynchronously
	 * @throws Exception
	 */
	private static void asyncInvokeBerockModelAndPutVectorsToS3(Document document, String fileName, String embeddingsResultFileName)
			throws Exception {
		var invocationARN= startAsyncInvokeBerockModel(document);
		
		while (true) {
			var gaiRequest = GetAsyncInvokeRequest.builder().invocationArn(invocationARN).build();
			var gaiResponse = BEDRDOCK_RUNTIME_CLIENT.getAsyncInvoke(gaiRequest);
			var status = gaiResponse.status();
			System.out.println("status: " + status);
			if (AsyncInvokeStatus.IN_PROGRESS.equals(status)) {
				Thread.sleep(20000);
			}
			if (AsyncInvokeStatus.COMPLETED.equals(status)) {
				var s3Uri = gaiResponse.outputDataConfig().s3OutputDataConfig().s3Uri();
				int i = 1;
				for (String line : new String(getS3ObjectWithEmbeddings(s3Uri,embeddingsResultFileName))
						.split("\\r?\\n|\\r")) {
					AsyncEmbeddingResponse asyncEmbeddingResponse = MAPPER.readValue(line,
							AsyncEmbeddingResponse.class);
					System.out.println("async embedding response: " + asyncEmbeddingResponse);
					putVectors(asyncEmbeddingResponse.embedding(), fileName + "_" + i);
					i++;
				}
				System.out.println("store s3 vector for " + fileName);
				return;
			}
		}
	}
	
	/** starts bedrock async invocation of the Bedrock Model and returns it invocations ARN 
	 * 
	 * @param document - document to be used as the model input
	 * @return invocation ARN
	 */
	private static String startAsyncInvokeBerockModel(Document document) {
		System.out.println("doc: " + document);
		var ais3dc = AsyncInvokeS3OutputDataConfig.builder().s3Uri(S3_EMBEDDINGS_DESTINATION_URI).build();
		var aiodc = AsyncInvokeOutputDataConfig.builder().s3OutputDataConfig(ais3dc).build();
		var saiRequest = StartAsyncInvokeRequest.builder().modelId(MODEL_ID).modelInput(document)
				.outputDataConfig(aiodc).build();
		System.out.println("saiReq: " + saiRequest);
		var saiResponse = BEDRDOCK_RUNTIME_CLIENT.startAsyncInvoke(saiRequest);
		var invocationARN = saiResponse.invocationArn();
		System.out.println("invocation ARN " + invocationARN);
		return invocationARN;
	}

	/**
	 * returns content of the file with the embeddings results
	 * @param s3Uri -s3 bucket URI
	 * @param embeddingsResultFileName - file name with embeddings result
	 * @return content of the file with the embeddings results
	 * @throws Exception
	 */
	private static byte[] getS3ObjectWithEmbeddings(String s3Uri, String embeddingsResultFileName) throws Exception {
		var s3UriModified = s3Uri.substring("s3://".length());
		var slashIndex = s3UriModified.indexOf("/");
		var s3Bucket = s3UriModified.substring(0, slashIndex);
		var key = s3UriModified.substring(slashIndex + 1) + "/" + embeddingsResultFileName;

		System.out.println("s3 uri " + s3Uri + " s3 bucket " + s3Bucket + " key " + key);
		var goRequest = GetObjectRequest.builder().bucket(s3Bucket).key(key).build();
		var goResponse = S3_CLIENT.getObject(goRequest);
		return goResponse.readAllBytes();
	}

	/**
	 * prepare and return Document to be used as the model input (will be converted
	 * to json automatically) 
	 * 
	 * @param s3_audio_uri - s3 URI with audio file
	 * @return prepare and return Document to be used as the model input
	 */
	private static Document prepareAudioDocument(String s3_audio_uri) {
		var s3locationConfig = Document.mapBuilder().putString("uri", s3_audio_uri).build();

		var sourceConfig = Document.mapBuilder().putDocument("s3Location", s3locationConfig).build();

		var durationConfig = Document.mapBuilder().putNumber("durationSeconds", 15).build();

		var audioConfig = Document.mapBuilder().putString("format", "mp3").putDocument("source", sourceConfig)
				.putDocument("segmentationConfig", durationConfig).build();

		var singleEmbeddingParams = Document.mapBuilder().putString("embeddingPurpose", "GENERIC_INDEX")
				.putNumber("embeddingDimension", EMBEDDING_DIMENSION).putDocument("audio", audioConfig).build();

		var request = Document.mapBuilder().putString("taskType", "SEGMENTED_EMBEDDING")
				.putDocument("segmentedEmbeddingParams", singleEmbeddingParams).build();

		return request;
	}

	/**
	 * prepares and returns Document to be used as the model input (will be converted
	 * to json automatically)
	 * 
	 * @param s3_video_uri - S3 URI with video file
	 * @return  Document to be used as the model input
	 */
	private static Document prepareVideoDocument(String s3_video_uri) {
		var s3locationConfig = Document.mapBuilder().putString("uri", s3_video_uri).build();

		var sourceConfig = Document.mapBuilder().putDocument("s3Location", s3locationConfig).build();

		var durationConfig = Document.mapBuilder().putNumber("durationSeconds", 15).build();

		var videoConfig = Document.mapBuilder().putString("format", "mp4")
				.putString("embeddingMode", "AUDIO_VIDEO_COMBINED").putDocument("source", sourceConfig)
				.putDocument("segmentationConfig", durationConfig).build();

		var singleEmbeddingParams = Document.mapBuilder().putString("embeddingPurpose", "GENERIC_INDEX")
				.putNumber("embeddingDimension", EMBEDDING_DIMENSION).putDocument("video", videoConfig).build();

		var request = Document.mapBuilder().putString("taskType", "SEGMENTED_EMBEDDING")
				.putDocument("segmentedEmbeddingParams", singleEmbeddingParams).build();

		return request;
	}

	/**
	 * put vector embeddings array into the S3 Vector store
	 *
	 * @param embeddings - array of embeddings
	 * @param key        - vector's key
	 */
	private static void putVectors(Float[] embeddings, String key) {
		var vd = VectorData.builder().float32(embeddings).build();
		var piv = PutInputVector.builder().data(vd).key(key)
				 //.metadata();
				.build();
		var pvRequest = PutVectorsRequest.builder().vectorBucketName(VECTOR_BUCKET).indexName(INDEX_NAME).vectors(piv)
				.build();
		S3_VECTORS_CLIENT.putVectors(pvRequest);
	}

	public static void main(String[] args) throws Exception {
		/*
		
		 createS3VectorBucketAndIndex();
		 
		 createAndStoreTextEmbeddings(AWS_LAMBDA_EMBEDDINGS,"AWS Lambda Definition");
		 createAndStoreTextEmbeddings(AZURE_FUNCTIONS__EMBEDDINGS,"Azure Functions Definition");
		 createAndStoreTextEmbeddings("Life is the most beautiful thing ever","Life  Definition");
		 
		 createAndStoreImageEmbeddings();
		 */
		 createAndStoreAudioEmbeddings();
		 /*
		 createAndStoreVideoEmbeddings();
		 
		 search("Azure Functions", 5);
		 search("AWS Lambda", 20);
		 
		 */
	}
}
