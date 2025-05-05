from datasets import load_dataset

from dense import dense_search
from hybrid import hybrid_search

from athina_tools import upload_dataset

dataset = load_dataset("PatronusAI/financebench", split="train")


# These are the docs for which we've uploaded the embeddings
docs = ['AMCOR_2022_8K_dated-2022-07-01', 'ADOBE_2017_10K', 'ACTIVISIONBLIZZARD_2019_10K', 'AES_2022_10K', 'AMAZON_2019_10K', '3M_2018_10K', '3M_2023Q2_10Q', 'AMCOR_2020_10K', '3M_2022_10K', 'AMAZON_2017_10K', 'ADOBE_2015_10K', 'ADOBE_2016_10K', "ADOBE_2022_10K"]

#final_data = {"query":[], "context":[], "type":[]}
final_data = []

i = 1
cnt = 1
for data in dataset:
    if data["doc_name"] in docs:
        if i==7:
            # df = pd.DataFrame(final_data)
            # df.to_csv(f"data{cnt}.csv")
            upload_dataset(f"rag_test_{cnt}", final_data)
            print(f"Uploaded rag_test_{cnt}")
            # final_data = {"query":[], "context":[], "type":[]}
            final_data = []
            cnt+=1
            i = 1

        question = data["question"]
        print(question)
        
        company = data["company"]
        year = data["doc_period"]

        dense_results = dense_search(question, company, year)
        dense_contents = [content for content,score in dense_results]

        final_data.append({
            "query":question,
            "context":dense_contents,
            "type":"dense"
        })

        # if dense_results:
        #     # print("\n🔍 **Search Results (Dense Search - PGVector):**\n")
        #     for idx, (content, score) in enumerate(dense_results, 1):
        #         dense_contents.append(f"{idx}. {content}\n\n")
        #         #print(f"{idx}. 📝 {content}...  (Similarity: {score:.4f})")
        # else:
        #     print("⚠️ No relevant results found.")

        hybrid_contents = hybrid_search(question, company, year)


        final_data.append({
            "query":question,
            "context":hybrid_contents,
            "type":"hybrid"
        })

        # if hybrid_results:
        #     # print("\n🔍 **Search Results (Dense Search - PGVector):**\n")
        #     for idx, (content, score) in enumerate(hybrid_results, 1):
        #         hybrid_contents.append(f"{idx}. {content}\n\n")
        #         #print(f"{idx}. 📝 {content}...  (Similarity: {score:.4f})")
        # else:
        #     print("⚠️ No relevant results found.")

        # answer = get_answer(question, "".join(hybrid_contents))
        # print(answer)

        i+=1
        print("\n#####################################\n")
    else:
        upload_dataset(f"rag_test_{cnt}", final_data)
        print(f"Uploaded rag_test_{cnt}")
        break


    