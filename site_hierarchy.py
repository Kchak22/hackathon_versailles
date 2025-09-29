import json

def extract_urls_from_jsonl(path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "url" in obj and isinstance(obj["url"], str):
                    urls.append(obj["url"])
            except json.JSONDecodeError:
                # ligne pas valide JSON, on ignore
                continue

    graph = {}
    for url in urls:
        if url.startswith("https://"):
            url = url[8:]
        elif url.startswith("http://"):
            url = url[7:]
        
        blocks = url.split("/")
        node = graph
        for block in blocks:
            if block == "":  # ignore vide
                continue
            if block not in node:
                node[block] = {}
            node = node[block]
    return graph


if __name__ == "__main__":
    graph = extract_urls_from_jsonl("versailles_semantic_complete_20250813_204248.jsonl")
    with open("site_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)