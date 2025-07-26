import os
from datetime import datetime
from typing import Dict, List, Any, Optional
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import re
import time
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import spacy

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_query_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InsuranceQueryProcessor:
    """Production-ready insurance claim processor with enhanced features"""
    
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables")
        
        try:
            # Initialize with better performance settings
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            
            self.embedder = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key,
                task_type="retrieval_document"
            )
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0,
                google_api_key=self.gemini_api_key,  # Fixed typo (was gemini_api_key)
                max_output_tokens=2048
            )
            
            # Enhanced VectorDB with metadata support
            from vectordb import InsuranceVectorDB
            self.db = InsuranceVectorDB(self.embedder)
            self.db.load("insurance_vector_db")
            
            self.parser = JsonOutputParser()
            self._initialize_prompts()
            
            # New: Cache for frequent queries
            self.query_cache = {}
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def _initialize_prompts(self):
        """Enhanced prompts with amount estimation and clause references"""
        self.query_expansion_prompt = ChatPromptTemplate.from_template("""
        As an insurance expert, expand this query for better search results:
        Original: {query}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "expanded_queries": ["list", "of", "queries"],
            "key_terms": ["list", "of", "important", "terms"],
            "potential_amount_ranges": {{
                "min": 0,
                "max": 0,
                "currency": "INR"
            }}
        }}
        """)
        
        self.decision_prompt = ChatPromptTemplate.from_template("""
        Analyze this insurance claim:
        {claim_details}
        
        Relevant policy clauses:
        {clauses}
        
        For each relevant clause, include:
        - Exact clause text
        - Source document
        - Applicability reasoning
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "decision": "approved|rejected|review",
            "reason": "detailed explanation with clause references",
            "confidence": "high|medium|low",
            "estimated_amount": {{
                "value": 0,
                "currency": "INR",
                "calculation_basis": "description"
            }},
            "clause_analysis": [
                {{
                    "id": "clause_id",
                    "text": "full clause text",
                    "source": "document_name.pdf",
                    "relevance": "high|medium|low",
                    "impact": "positive|negative|neutral"
                }}
            ],
            "next_steps": ["list", "of", "actions"]
        }}
        """)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Enhanced processing with caching and detailed output"""
        try:
            # Check cache first
            if query in self.query_cache:
                logger.info(f"Returning cached result for query: {query[:50]}...")
                return self.query_cache[query]
            
            details = self._extract_claim_details(query)
            if not details.get("procedure"):
                return {"error": "No medical procedure identified"}
            
            # Enhanced query expansion with amount estimation
            expanded = self._generate_queries(query)
            queries = expanded.get("expanded_queries", [query])
            amount_estimate = expanded.get("potential_amount_ranges", {})
            
            clauses = self._retrieve_clauses(queries)
            if not clauses:
                return {
                    "decision": "review",
                    "reason": "No matching clauses found",
                    "confidence": "low",
                    "clause_analysis": []
                }
            
            # Enhanced decision with full clause details
            decision = self._make_decision(details, clauses)
            decision["amount_estimate"] = amount_estimate
            decision["timestamp"] = datetime.now().isoformat()
            
            # Cache the result
            self.query_cache[query] = decision
            return decision
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": "Processing failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_claim_details(self, query: str) -> Dict[str, Any]:
        """Enhanced extraction with amount detection"""
        doc = self.nlp(query)
        
        # Enhanced patterns
        age_gender = re.search(r'(\d+)\s*(year old|yo|yr)?\s*(male|female|M|F)\b', query, re.I)
        procedure = re.search(r'(knee surgery|[\w\s]+ surgery|[\w\s]+ treatment|[\w\s]+ bypass)', query, re.I)
        policy_age = re.search(r'(\d+)\s*(month|year|day)s?\s*old', query, re.I)
        amount = re.search(r'(?:Rs\.?|INR)\s*(\d+[,\.]\d+)', query, re.I)
        
        return {
            "age_gender": age_gender.group(0) if age_gender else "",
            "procedure": procedure.group(1).strip() if procedure else "",
            "policy_age": self._parse_policy_age(policy_age) if policy_age else 0,
            "claimed_amount": float(amount.group(1).replace(',', '')) if amount else None,
            "raw_query": query,
            "extraction_confidence": "high" if procedure and age_gender else "medium"
        }

    def _parse_policy_age(self, match) -> int:
        """Convert policy age to months"""
        if not match:
            return 0
        num = int(match.group(1))
        unit = match.group(2).lower()
        return num * 12 if 'year' in unit else num

    def _generate_queries(self, query: str) -> Dict[str, Any]:
        """Generate search queries with amount estimation"""
        try:
            chain = self.query_expansion_prompt | self.llm | self.parser
            response = chain.invoke({"query": query})
            return {
                "expanded_queries": response.get("expanded_queries", [query]),
                "key_terms": response.get("key_terms", []),
                "potential_amount_ranges": response.get("potential_amount_ranges", {
                    "min": 0,
                    "max": 0,
                    "currency": "INR"
                })
            }
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return {
                "expanded_queries": [query],
                "key_terms": [],
                "potential_amount_ranges": {
                    "min": 0,
                    "max": 0,
                    "currency": "INR"
                }
            }

    def _retrieve_clauses(self, queries: List[str]) -> List[Dict]:
        """Retrieve clauses with simplified interface"""
        clauses = []
        for query in queries:
            try:
                results = self.db.search(query=query, k=3)
                clauses.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {str(e)}")
        return clauses

    def _make_decision(self, details: Dict, clauses: List[Dict]) -> Dict:
        """Enhanced decision with full clause analysis"""
        try:
            chain = self.decision_prompt | self.llm | self.parser
            response = chain.invoke({
                "claim_details": details,
                "clauses": clauses
            })
            
            # Post-process to add source documents
            for clause in response.get("clause_analysis", []):
                clause_id = clause.get("id")
                if clause_id:
                    clause["source"] = self._get_clause_source(clause_id)
            
            return response
        except Exception as e:
            logger.error(f"Decision failed: {str(e)}")
            return {
                "decision": "review",
                "reason": "Decision process failed",
                "confidence": "low",
                "clause_analysis": [],
                "error_details": str(e)
            }

    def _get_clause_source(self, clause_id: str) -> Optional[str]:
        """Helper to get source document for a clause"""
        # Implement your document lookup logic here
        return f"policy_document_{clause_id[:3]}.pdf"

if __name__ == "__main__":
    processor = InsuranceQueryProcessor()
    test_queries = [
        "46M, knee surgery (Rs. 2,50,000), Pune, 3-month policy",
        "33F, cosmetic surgery, Mumbai, 1-year policy (Max cover: 5L)", 
        "60M, emergency heart bypass, Delhi, 6-month policy"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\nProcessing: {query}")
        start_time = time.time()
        result = processor.process_query(query)
        elapsed = time.time() - start_time
        
        # Enhanced output formatting
        print(f"\n{' QUERY RESULT '.center(80, '=')}")
        print(f"Query: {query}")
        print(f"Processing time: {elapsed:.2f}s")
        print("\nDecision Summary:")
        print(f"Status: {result.get('decision', 'N/A').upper()}")
        print(f"Reason: {result.get('reason', 'No reason provided')}")
        
        if 'clause_analysis' in result:
            print("\nRelevant Clauses:")
            for clause in result['clause_analysis']:
                print(f"\n• {clause.get('id')} ({clause.get('relevance', 'N/A')} relevance)")
                print(f"  Source: {clause.get('source', 'Unknown')}")
                print(f"  Impact: {clause.get('impact', 'N/A')}")
                print(f"  Excerpt: {clause.get('text', '')[:100]}...")
        
        if 'amount_estimate' in result:
            amt = result['amount_estimate']
            print(f"\nEstimated Amount: {amt.get('currency', 'INR')} {amt.get('min', 0)}-{amt.get('max', 0)}")
        
        results.append({
            "query": query,
            "response": result,
            "processing_time": elapsed,
            "timestamp": datetime.now().isoformat()
        })
    
    # Enhanced output saving
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"insurance_results_{timestamp}.json")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "system": "InsuranceQueryProcessor",
                "version": "1.1",
                "run_date": timestamp
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_filename}")
    print(f"Total queries processed: {len(results)}")
    print(f"Average processing time: {sum(r['processing_time'] for r in results)/len(results):.2f}s")