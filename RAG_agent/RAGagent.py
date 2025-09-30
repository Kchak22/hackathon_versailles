from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import langchain_core.documents
from pathlib import Path
import json, os



class RAGagent():

    def __init__(self):
        """
        Initialize the embedding model and load the hierarchical infos (group - subgroup - place - object)
        """
        self.embedding_model_name : str = "sentence-transformers/all-MiniLM-L12-v2"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        with open(os.path.join("RAG_agent","hierarchy.json"), "r", encoding="utf-8") as f:
            self.hierarchy = json.load(f)
        
        self.build_index()

  
    def build_index(self):
        """
        Build one faiss index where each chunk represents an object from the hierarchy file 
        """
        chunks = []
        for group in self.hierarchy:
            group_id = self.hierarchy[group]["group_id"]
            group_title = self.hierarchy[group]["title"]
            for subgroup in self.hierarchy[group]["subgroups"]:
                subgroup_id = subgroup["subgroup_id"]
                subgroup_title = subgroup["title"]
                for place in subgroup["places"]:
                    place_id = place["place_id"]
                    place_title = place["title"]
                    for object in place["objects"]:
                        object_id = object["object_id"]
                        object_title = object["title"]
                        content = object["text"] + '\n' + '\n'.join(object["important_facts"])
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


    def search_objects(self, interests : list[str], k : int = 5) : 
        """
        Return metadata of k top ojects for each interest of the user. The pertinence of the object based on the interest is given by a score which can be obtained by res[interest]["score"].  
        Args : 
            - interests (list[str]): a list of the user interests.
            - k : the number of pertinent objects to return for a given interest.
        """
        faiss_path = os.path.join("checkpoints", "faiss_index")
        faiss_store = FAISS.load_local(faiss_path, self.embedding_model, allow_dangerous_deserialization=True)

        res = {}
        for interest in interests:
            query = interest
            retrieved = faiss_store.similarity_search_with_score(query, k=k)
            enriched = []
            for doc, score in retrieved:
                meta = doc.metadata.copy()
                meta["score"] = float(score)  
                enriched.append(meta)
            res[interest] = enriched
        return res


if __name__ == "__main__": 
    ragAgent = RAGagent()
    res = ragAgent.search_objects(["mythologie"])
    for interest in res :
        print(interest)
        print('\n')
        for object in res[interest]:
            print(object)
            print("\n")