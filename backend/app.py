from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from datetime import datetime
import json

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HSCodeSBERTMapper:
    def __init__(self, hs_file_path, compliance_file_path, model_name='all-mpnet-base-v2'):
        self.hs_file_path = hs_file_path
        self.compliance_file_path = compliance_file_path
        self.df_hs = self._load_and_prepare_hs_data()
        self.df_compliance = self._load_and_prepare_compliance_data()
        
        print(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print("Creating embeddings for all HS Codes...")
        self.hs_embeddings = self.model.encode(
            self.df_hs['description'].tolist(),
            show_progress_bar=True
        )
        print("HS Embedding complete.")
        
        print("Creating embeddings for compliance keywords...")
        self.compliance_keywords = self.df_compliance[
            self.df_compliance['keyword'].notna() & (self.df_compliance['keyword'] != '')
        ]['keyword'].tolist()
        if self.compliance_keywords:
            self.compliance_embeddings = self.model.encode(
                self.compliance_keywords,
                show_progress_bar=True
            )
        else:
            self.compliance_embeddings = np.array([])
        print("Compliance Embedding complete.")

    def _load_and_prepare_hs_data(self):
        df = pd.read_csv(self.hs_file_path)
        if 'Unnamed: 3' in df.columns:
            df = df.drop(columns=['Unnamed: 3'])
        df.columns = ['hs_code', 'description', 'import_duty']
        df['import_duty'] = df['import_duty'].str.replace(',', '.')
        return df

    def _load_and_prepare_compliance_data(self):
        df = pd.read_csv(self.compliance_file_path)
        return df

    def extract_item_descriptions(self, sample_text):
        lines = sample_text.split('\n')
        descriptions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Handle "Item: description" format
            match = re.search(r'item\s*:\s*(.*)', line, re.IGNORECASE)
            if match:
                descriptions.append(match.group(1).strip())
            # Also handle plain text without "Item:" prefix
            elif line and not line.lower().startswith('item'):
                descriptions.append(line)
        return descriptions

    def match_hs_code(self, raw_description):
        invoice_embedding = self.model.encode([raw_description])
        similarity_scores = cosine_similarity(invoice_embedding, self.hs_embeddings).flatten()
        best_match_index = np.argmax(similarity_scores)
        best_score = similarity_scores[best_match_index]

        special_handling_keywords = [
            ('shrimp|prawn', 'fresh|frozen'),
            ('fish', 'fresh|live'),
            ('poultry', 'live')
        ]
        
        for product_keyword, state_keyword in special_handling_keywords:
            if (re.search(product_keyword, raw_description.lower()) and 
                re.search(state_keyword, raw_description.lower())):
                special_candidates = self.df_hs[
                    self.df_hs['description'].str.lower().str.contains(product_keyword) &
                    self.df_hs['description'].str.lower().str.contains(state_keyword)
                ]
                if not special_candidates.empty:
                    candidate_embeddings = self.model.encode(special_candidates['description'].tolist())
                    candidate_scores = cosine_similarity(invoice_embedding, candidate_embeddings).flatten()
                    best_candidate_index = np.argmax(candidate_scores)
                    best_candidate_score = candidate_scores[best_candidate_index]
                    if best_candidate_score > 0.30:
                        match_row = special_candidates.iloc[best_candidate_index]
                        return {
                            'hs_code': match_row['hs_code'],
                            'hs_description': match_row['description'],
                            'import_duty': match_row['import_duty'],
                            'score': float(best_candidate_score)
                        }

        if best_score > 0.30:
            match_row = self.df_hs.iloc[best_match_index]
            return {
                'hs_code': match_row['hs_code'],
                'hs_description': match_row['description'],
                'import_duty': match_row['import_duty'],
                'score': float(best_score)
            }
        else:
            return {
                'hs_code': 'N/A',
                'hs_description': 'No strong match found',
                'import_duty': 'N/A',
                'score': float(best_score)
            }

    def match_compliance_rules(self, hs_code, raw_description, hs_description):
        matching_rules = []
        df_filtered = self.df_compliance[self.df_compliance['country'] == 'AE']

        if hs_code != 'N/A':
            hs_str = str(hs_code)
            prefix_matches = df_filtered[df_filtered['hs_prefix'].notna() & (df_filtered['hs_prefix'] != '')]
            for _, row in prefix_matches.iterrows():
                if hs_str.startswith(str(row['hs_prefix'])):
                    matching_rules.append({
                        'requirement_code': row['requirement_code'],
                        'requirement_name': row['requirement_name'],
                        'agency': row['agency'],
                        'details': row['details'],
                        'severity': row['severity'],
                        'source': row['source']
                    })
        
        description_combined = f"{raw_description} {hs_description}".lower()

        keyword_matches = df_filtered[df_filtered['keyword'].notna() & (df_filtered['keyword'] != '')]
        for _, row in keyword_matches.iterrows():
            keyword = row['keyword'].lower()
            if keyword in description_combined:
                matching_rules.append({
                    'requirement_code': row['requirement_code'],
                    'requirement_name': row['requirement_name'],
                    'agency': row['agency'],
                    'details': row['details'],
                    'severity': row['severity'],
                    'source': row['source']
                })
        
        if self.compliance_embeddings.size > 0:
            invoice_embedding = self.model.encode([description_combined])
            similarity_scores = cosine_similarity(invoice_embedding, self.compliance_embeddings).flatten()
            best_keyword_index = np.argmax(similarity_scores)
            best_keyword_score = similarity_scores[best_keyword_index]
            if best_keyword_score > 0.25:
                best_keyword = self.compliance_keywords[best_keyword_index]
                semantic_matches = df_filtered[df_filtered['keyword'] == best_keyword]
                for _, row in semantic_matches.iterrows():
                    matching_rules.append({
                        'requirement_code': row['requirement_code'],
                        'requirement_name': row['requirement_name'],
                        'agency': row['agency'],
                        'details': row['details'],
                        'severity': row['severity'],
                        'source': row['source']
                    })
        
        matching_rules = [dict(t) for t in {tuple(d.items()) for d in matching_rules}]
        return matching_rules

    def process_invoice(self, sample_text):
        descriptions = self.extract_item_descriptions(sample_text)
        results = []
        for raw_desc in descriptions:
            hs_match = self.match_hs_code(raw_desc)
            compliance_rules = self.match_compliance_rules(
                hs_match['hs_code'], 
                raw_desc, 
                hs_match['hs_description']
            )
            results.append({
                'invoice_description': raw_desc,
                'hs_code': hs_match['hs_code'],
                'hs_description': hs_match['hs_description'],
                'import_duty': hs_match['import_duty'],
                'score': round(hs_match['score'], 2),
                'compliance_rules': compliance_rules
            })
        
        return results

# Initialize the mapper once when the server starts
print("Initializing HS Code Mapper...")
mapper = HSCodeSBERTMapper("hs_code_detail.csv", "compliance_rule_book.csv")
print("Mapper ready!")

class ProductRequest(BaseModel):
    products: str

class ProductResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_products: int

@app.get("/")
def read_root():
    return {"message": "HS Code Mapper API is running", "status": "ok"}

@app.post("/api/process", response_model=ProductResponse)
async def process_products(request: ProductRequest):
    try:
        if not request.products.strip():
            raise HTTPException(status_code=400, detail="No products provided")
        
        results = mapper.process_invoice(request.products)
        
        # Save to CSV for record keeping
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f'results_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        
        return {
            "results": results,
            "total_products": len(results)
        }
    except Exception as e:
        print(f"Error processing products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": mapper is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)