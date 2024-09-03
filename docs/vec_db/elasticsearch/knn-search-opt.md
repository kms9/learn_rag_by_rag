## knn Optional


> (Optional, object or array of objects) Defines the kNN query to run.

Properties of knn object

###  field
(Required, string) The name of the vector field to search against. Must be a dense_vector field with indexing enabled.

###  filter
(Optional, Query DSL object) Query to filter the documents that can match. The kNN search will return the top k documents that also match this filter. The value can be a single query or a list of queries. If filter is not provided, all documents are allowed to match.

### k
(Optional, integer) Number of nearest neighbors to return as top hits. This value must be less than or equal to num_candidates. Defaults to size.

### num_candidates
(Optional, integer) The number of nearest neighbor candidates to consider per shard. Needs to be greater than k, or size if k is omitted, and cannot exceed 10,000. Elasticsearch collects num_candidates results from each shard, then merges them to find the top k results. Increasing num_candidates tends to improve the accuracy of the final k results. Defaults to Math.min(1.5 * k, 10_000).

###  query_vector
(Optional, array of floats) Query vector. Must have the same number of dimensions as the vector field you are searching against. Must be either an array of floats or a hex-encoded byte vector.

### query_vector_builder
(Optional, object) A configuration object indicating how to build a query_vector before executing the request. You must provide a query_vector_builder or query_vector, but not both. Refer to Perform semantic search to learn more.

###  similarity
(Optional, float) The minimum similarity required for a document to be considered a match. The similarity value calculated relates to the raw similarity used. Not the document score. The matched documents are then scored according to similarity and the provided boost is applied.

The similarity parameter is the direct vector similarity calculation.

-  l2_norm: also known as Euclidean, will include documents where the vector is within the dims dimensional hypersphere with radius similarity with origin at query_vector.
- cosine, dot_product, and max_inner_product: Only return vectors where the cosine similarity or dot-product are at least the provided similarity.
Read more here: knn similarity search