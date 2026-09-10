# Data Processing

This folder contains the raw and processed data used for the Hiver Support Agent project.

## Dataset

We used the **Customer Support on Twitter** dataset from Kaggle (`thoughtvector/customer-support-on-twitter`). The dataset was loaded into pandas and analyzed using `author_id` to identify brands with high support activity.

### Brand Selection

The highest-volume brands were shortlisted and approximately 50 random inbound tweets from each were manually inspected for issue quality and relevance. We selected **AmazonHelp** because it had the highest volume and contained diverse, recognizable customer-support issues such as delivery problems, missing orders, refunds, returns, payments, and product-related issues.

### Filtering & Thread Reconstruction

The dataset was filtered to AmazonHelp tweets. Tweets with a valid `in_response_to_tweet_id` were retained to connect AmazonHelp replies with their parent customer tweets. Incomplete interactions were removed, and duplicate interactions were eliminated where applicable.

### Subsampling

To keep the dataset manageable while preserving a representative sample, **30,000 interactions** were randomly selected using `random_state=42`.

| Stage             |       Rows |
| ----------------- | ---------: |
| AmazonHelp tweets |    169,840 |
| With parent tweet |    169,287 |
| After cleaning    |    168,823 |
| Final sample      | **30,000** |

The final processed dataset is stored as:

`processed/amazonhelp_threads.parquet`

The `processed` dataset contains the customer tweet, AmazonHelp reply, tweet IDs, and response relationships required for subsequent retrieval and support-agent development.
