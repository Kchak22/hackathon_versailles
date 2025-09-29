from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import langchain_core.documents
import json
from pathlib import Path



class RAGagent():

    def __init__(self):
        self.embedding_model_name : str = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        with open("site_hierarchy.json", "r", encoding="utf-8") as f:
            self.site_hierarchy = json.load(f)

        with open("hierarchy.json", "r", encoding="utf-8") as f:
            self.hierarchy = json.load(f)

        self.parsed_links = {}
        with open("versailles_semantic_complete_20250813_204248.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "url" in obj and isinstance(obj["url"], str):
                        self.parsed_links[obj["url"]] = obj
                except json.JSONDecodeError:
                    # ligne pas valide JSON, on ignore
                    continue
        


    # def build_index(self):
    #     # Building Faiss index 
    #     chunks = []
    #     for domain in self.site_hierarchy["www.chateauversailles.fr"]["decouvrir"]["domaine"] :
    #         subdomains = self.site_hierarchy["www.chateauversailles.fr"]["decouvrir"]["domaine"][domain]
    #         if subdomains : 
    #             for subdomain in subdomains:
    #                 url = "https://www.chateauversailles.fr/decouvrir/domaine/" + domain + '/' + subdomain
    #                 parsed_link = self.parsed_links[url]
    #                 content = '\n'.join([box.get("text", "") for box in parsed_link["content"]])
    #                 chunk = langchain_core.documents.Document(
    #                     page_content= content,
    #                     metadata={"url": url, "title" : parsed_link["title"]}
    #                 )
    #                 chunks.append(chunk)
    #                 print(f"✅ Chunk created for link: {url}")

    #         else: 
    #             url = "https://www.chateauversailles.fr/decouvrir/domaine/" + domain  
    #             parsed_link = self.parsed_links[url]
    #             content = '\n'.join([box.get("text", "") for box in parsed_link["content"]])
    #             chunk = langchain_core.documents.Document(
    #                 page_content= content,
    #                 metadata={"url": url, "title" : parsed_link["title"]}
    #             )
    #             chunks.append(chunk)
    #             print(f"✅ Chunk created for link: {url}")

    #     save_dir = Path("checkpoints")
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     faiss_store = FAISS.from_documents(chunks, self.embedding_model)
    #     faiss_dir = save_dir / "faiss_index"
    #     faiss_store.save_local(str(faiss_dir))
    #     print(f"💾 FAISS index saved to: {faiss_dir}")


    # def search_places(self, interests : list[str]) : 
    #     faiss_path = "checkpoints/faiss_index"
    #     faiss_store = FAISS.load_local(faiss_path, self.embedding_model, allow_dangerous_deserialization=True)

    #     res = {}
    #     for interest in interests:
    #         query = f"Les places liés à {interest}"
    #         retrieved_chunks = faiss_store.similarity_search(query, k=5)
    #         res[interest] = [{"url" : chunk.metadata["url"], "title" : chunk.metadata["title"]} for chunk in retrieved_chunks]
    #     return res

    def build_index(self):
        # Building Faiss index 
        chunks = []
        group_id = self.hierarchy["group_id"]
        group_title = self.hierarchy["title"]
        for subgroup in self.hierarchy["subgroups"]:
            subgroup_id = subgroup["subgroup_id"]
            subgroup_title = subgroup["title"]
            for place in subgroup["places"]:
                place_id = place["place_id"]
                place_title = place["title"]
                for object in place["objects"]:
                    object_id = object["object_id"]
                    object_title = object["title"]
                    content = object["text"]
                    chunk = langchain_core.documents.Document(
                            page_content= content,
                            metadata={
                                "group_id" : group_id,
                                "group_title" : group_title,
                                "subgroup_id" : subgroup_id,
                                "subgroup_title" : subgroup_title,
                                "place_id" : place_id,
                                "place_title" : place_title,
                                "object_id" : object_id,
                                "object_title" : object_title,
                            }
                        )
                    chunks.append(chunk)
                    print(f"✅ Chunk created for object: {object_title}")

        save_dir = Path("checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss_store = FAISS.from_documents(chunks, self.embedding_model)
        faiss_dir = save_dir / "faiss_index"
        faiss_store.save_local(str(faiss_dir))
        print(f"💾 FAISS index saved to: {faiss_dir}")


    def search_places(self, interests : list[str]) : 
        faiss_path = "checkpoints/faiss_index"
        faiss_store = FAISS.load_local(faiss_path, self.embedding_model, allow_dangerous_deserialization=True)

        res = {}
        for interest in interests:
            query = f"Les places liés à {interest}"
            retrieved_chunks = faiss_store.similarity_search(query, k=5)
            res[interest] = [chunk.metadata for chunk in retrieved_chunks]
        return res


if __name__ == "__main__": 
    ragAgent = RAGagent()
    # ragAgent.build_index()
    res = ragAgent.search_places(["mythologie"])
    for interest in res :
        print(interest)
        print('\n')
        for object in res[interest]:
            print(object)
            print("\n")