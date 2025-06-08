import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from dataclasses import dataclass
from typing_extensions import TypedDict
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from google import genai
from google.genai import types
import os, json
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, HTTPException
import uvicorn
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for name in logging.root.manager.loggerDict:
    if name != "__main__":
        logging.getLogger(name).setLevel(logging.WARNING)

# Database configuration
with open('config.json', 'r') as f:
    DB_CONFIG = json.load(f)

# State definition for LangGraph
class AgentState(TypedDict):
    customer_id: str
    customer_profile: Optional[Dict]
    purchase_patterns: Optional[Dict]
    product_affinities: Optional[List[Dict]]
    opportunity_scores: Optional[List[Dict]]
    research_report: Optional[str]
    recommendations: Optional[List[Dict]]
    error: Optional[str]

@dataclass
class CustomerProfile:
    customer_id: str
    customer_name: str
    industry: str
    annual_revenue: int
    employees: int
    priority: str
    rating: str
    account_type: str
    location: str
    current_products: str
    product_usage: int
    cross_sell_synergy: str
    last_activity: str
    opportunity_stage: str

class DatabaseManager:
    def __init__(self, config: Dict):
        self.config = config
    
    def get_connection(self):
        return psycopg2.connect(**self.config)
    
    def execute_query(self, query: str, params: tuple = None):
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    try:
                        return cursor.fetchall()
                    except psycopg2.ProgrammingError:
                        # If no rows are returned, fetchall will raise ProgrammingError
                        return []
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise


class CustomerContextAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def extract_customer_profile(self, state: AgentState) -> AgentState:
        try:
            customer_id = state["customer_id"]
            logger.info(f"Extracting profile for customer: {customer_id}")
            
            # Get customer basic info
            customer_query = """SELECT DISTINCT customer_id, customer_name, industry, annual_revenue_usd, number_of_employees, customer_priority, customer_rating, account_type, "location", current_products, product_usage_percent, cross_sell_synergy, last_activity_date, opportunity_stage FROM customer_data WHERE customer_id = %s"""
            
            customer_data = self.db_manager.execute_query(customer_query, (customer_id,))
            
            if not customer_data:
                state["error"] = f"Customer {customer_id} not found"
                return state
            
            customer = customer_data[0]
            customer["annual_revenue_usd"] = round(float(customer["annual_revenue_usd"][1:].replace(",", "").strip()))
            
            purchase_query = """
            SELECT product, quantity, unit_price_usd, total_price_usd, purchase_date
            FROM customer_data 
            WHERE customer_id = %s
            ORDER BY purchase_date DESC
            """
            
            purchases = self.db_manager.execute_query(purchase_query, (customer_id,))
            
            for p in purchases:
                p["total_price_usd"] = round(float(p["total_price_usd"][1:].replace(",", "").strip()))
                p["unit_price_usd"] = round(float(p["unit_price_usd"][1:].replace(",", "").strip()))
            
            profile = {
                "customer_id": customer["customer_id"],
                "customer_name": customer["customer_name"],
                "industry": customer["industry"],
                "annual_revenue": customer["annual_revenue_usd"],
                "employees": customer["number_of_employees"],
                "priority": customer["customer_priority"],
                "rating": customer["customer_rating"],
                "account_type": customer["account_type"],
                "location": customer["location"],
                "current_products": customer["current_products"],
                "product_usage": customer["product_usage_percent"],
                "cross_sell_synergy": customer["cross_sell_synergy"],
                "last_activity": customer["last_activity_date"],
                "opportunity_stage": customer["opportunity_stage"],
                "purchase_history": [dict(p) for p in purchases],
                "total_purchases": len(purchases),
                "total_spent": sum(p["total_price_usd"] for p in purchases),
            }
            
            state["customer_profile"] = profile
            logger.info(f"Successfully extracted profile for {customer['customer_name']}")
            
        except Exception as e:
            logger.error(f"Error in CustomerContextAgent: {e}")
            state["error"] = str(e)
        
        return state

class PurchasePatternAnalysis:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def analyze_patterns(self, state: AgentState) -> AgentState:
        try:
            if state.get("error") or not state.get("customer_profile"):
                return state
            
            customer_profile = state["customer_profile"]
            purchase_history = customer_profile["purchase_history"]
            
            # Analyze purchase frequency
            product_purchase_freq = Counter(p["product"] for p in purchase_history)
            product_spending = defaultdict(float)
            
            for purchase in purchase_history:
                product_spending[purchase["product"]] += purchase["total_price_usd"]
            
            # Missing Oppotunity
            product_list = list(product_purchase_freq.keys())
            placeholders = ', '.join(['%s'] * len(product_list))
            
            missing_products_query = f"""SELECT DISTINCT product FROM customer_data WHERE product NOT IN ({placeholders})"""
            missing_products = [p['product'] for p in self.db_manager.execute_query(missing_products_query, product_list)]

            patterns = {
                "frequent_products": dict(product_purchase_freq.most_common(5)),  # Top 5 frequent products
                "product_spending": dict(product_spending),
                "missing_opportunities": missing_products,
                "purchase_frequency_score": round(len(purchase_history) / 12, 3) if purchase_history else 0,  # frequency score calculated per month
                "avg_order_value": (sum(p["total_price_usd"] for p in purchase_history) / len(purchase_history)) if purchase_history else 0
            }
            
            state["purchase_patterns"] = patterns
            logger.info(f"Analyzed purchase patterns - {len(missing_products)} missing opportunities found")
        
        except Exception as e:
            logger.error(f"Error in PurchasePatternAnalysis: {e}")
            state["error"] = str(e)
        
        return state

class ProductAffinityAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def suggest_related_products(self, state: AgentState) -> AgentState:
        try:
            affinites = {}
            if state.get("error") or not state.get("customer_profile"):
                return state
            
            customer_profile = state["customer_profile"]
            product_purchase_freq = Counter(p["product"] for p in customer_profile['purchase_history'])
            product_list = list(product_purchase_freq.keys())
            placeholders = ', '.join(['%s'] * len(product_list))
            
            # Cross-sell opportunities
            related_products_query = f"""SELECT DISTINCT product, COUNT(*) AS product_count FROM customer_data WHERE customer_id = ANY(SELECT DISTINCT customer_id FROM customer_data WHERE product IN ({placeholders})) AND product NOT IN ({placeholders}) GROUP BY product ORDER BY product_count DESC"""
            query_result = self.db_manager.execute_query(related_products_query, (product_list+ product_list))
            if query_result is None or query_result == []:
                    state["error"] = "No cross-sell opportunities found"
            else:         
                related_products = {p['product']: p['product_count'] for p in query_result}
                affinites['cross-sell'] = related_products
                logger.info(f"Analyzed product affinities - {len(related_products)} cross-sell opportunities found")
            
            # Upsell opportunities
            upsell_query = f"""SELECT product, (AVG(total_price)::numeric::money) as avg_price FROM (SELECT customer_id, product, SUM(total_price_usd::numeric::float8) AS total_price FROM customer_data WHERE product IN ({placeholders}) AND customer_id != %s GROUP BY customer_id, product) GROUP BY product ORDER BY product"""
            query_result = self.db_manager.execute_query(upsell_query, (product_list+[customer_profile["customer_id"]]))
            if query_result is None or query_result == []:
                state["error"] = "No upsell opportunities found"
            else:
                upsell_products = {p['product']: (round(float(p["avg_price"][1:].replace(",", "").strip()), 2)) for p in query_result}

                upsell_opportunities = {}
                for product, customer_value in state['purchase_patterns']['product_spending'].items():
                    if product in upsell_products:
                        if(customer_value > upsell_products[product]*1.2):
                            upsell_opportunities[product] = (customer_value - upsell_products[product])/ upsell_products[product]*100       # Percentage the customer is spending per product compared to average
                affinites["upsell"] = upsell_opportunities
                logger.info(f"Analyzed product affinities - {len(related_products)} upsell opportunities found")
            state["product_affinities"] = affinites
            
        except Exception as e:
            logger.error(f"Error in ProductAffinityAgent: {e}")
            state["error"] = str(e)
        
        return state

class OpportunityScoringAgent:
    def score_opportunities(self, state: AgentState) -> AgentState:
        if state.get("error") or not state.get("product_affinities"):
            return state
        # Cross-sell scoring
        total = sum(state['product_affinities']['cross-sell'].values())
        scored_cross_sell = {
            product: (qty / total) * 100
            for product, qty in state['product_affinities']['cross-sell'].items()
        }
        
        sorted_cross_sell = dict(sorted(scored_cross_sell.items(), key=lambda x: x[1], reverse=True))

        # Upsell scoring
        # High spend over peers - low upsell, low spend over peers - high upsell
        upsell_scores = {
            product: max(0, 100 - diff)
            for product, diff in state['product_affinities']['upsell'].items()
        }
        
        upsell_scores = dict(sorted(upsell_scores.items(), key=lambda x: x[1], reverse=True))
        
        scores = {
            "cross_sell": sorted_cross_sell,
            "cross_sell_reasoning": "Cross-sell opportunities are scored based on the frequency of purchases of related products. Values are percentages, higher values indicate stronger cross-sell potential.",
            "upsell": upsell_scores,
            "upsell_reasoning": "Upsell opportunities are scored based on the difference between the customer's spending and the average spending of peers on the same product. Higher scores indicate greater potential for upsell.",
        }
        state["opportunity_scores"] = scores
        logger.info(f"Scored opportunities - {len(scores['cross_sell'])} cross-sell and {len(scores['upsell'])} upsell opportunities remain after filtering")
        return state

class RecommendationReportAgent:
    def generate_report(self, state: AgentState) -> AgentState:
        
        format = """Research Report: Cross-Sell and Upsell Opportunities for Acme Corp
Introduction:
This report analyzes recent purchasing behavior of Acme Corp and benchmarks against
industry peers to identify cross-sell and upsell opportunities.
Customer Overview:
- Industry: Construction
- Annual Revenue: $1.5M
- Recent Purchases: Generators, Drills
Data Analysis:
- Purchase patterns indicate frequent purchases of generators, but limited spending on
accessory products.
- Industry peers commonly purchase backup batteries and safety gear.
- Product affinity analysis suggests drill bits and protective gloves as complementary
products.
Recommendations:
1. Offer Backup Batteries as a cross-sell opportunity for Generators.
2. Promote Drill Bits and Protective Gloves as upsell items to existing Drill owners.
3. Explore Heavy Equipment Rental services popular in the construction sector.
Conclusion:
Targeted cross-sell and upsell campaigns focusing on these products can increase revenue
and customer satisfaction."""
        
        client = genai.Client(api_key = os.getenv("Gemini_API_Key"))
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash-8b",
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert business and data analyst. Generate a comprehensive report in the given example format based on the provided data (customer data and analysis results). Do not output anything outside of this format.",
                ),
                contents=f"REPORT FORMAT: {format}\nDATA: {state}"
            )
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            state["error"] = str(e)
            return state
        logger.info("Recommendation report generated successfully")
        
        self._create_report_file(response.text)
        
        state["research_report"] = response.text
        return state
    
    def _create_report_file(self, report):
        with open("recommendation_report.txt", "w") as file:
            file.write(report)
        logger.info("Recommendation report saved to recommendation_report.txt")

class CrossSellUpsellAgent:
    def __init__(self, db_config: Dict):
        self.db_manager = DatabaseManager(db_config)
        self.customer_agent = CustomerContextAgent(self.db_manager)
        self.pattern_agent = PurchasePatternAnalysis(self.db_manager)
        self.affinity_agent = ProductAffinityAgent(self.db_manager)
        self.scoring_agent = OpportunityScoringAgent()
        self.report_agent = RecommendationReportAgent()
        
        # Build the graph
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("customer_context_node", self.customer_agent.extract_customer_profile)
        workflow.add_node("purchase_patterns_node", self.pattern_agent.analyze_patterns)
        workflow.add_node("product_affinity_node", self.affinity_agent.suggest_related_products)
        workflow.add_node("opportunity_scoring_node", self.scoring_agent.score_opportunities)
        workflow.add_node("generate_report_node", self.report_agent.generate_report)
        
        # Define the flow
        workflow.set_entry_point("customer_context_node")
        workflow.add_edge("customer_context_node", "purchase_patterns_node")
        workflow.add_edge("purchase_patterns_node", "product_affinity_node")
        workflow.add_edge("product_affinity_node", "opportunity_scoring_node")
        workflow.add_edge("opportunity_scoring_node", "generate_report_node")
        workflow.add_edge("generate_report_node", END)
        
        return workflow.compile()
    
    def run_analysis(self, customer_id: str) -> Dict[str, Any]:
        initial_state = AgentState(customer_id=customer_id)
        result = self.workflow.invoke(initial_state)
        
        if result.get("error"):
            raise Exception(result["error"])
        
        return {
            "research_report": result.get("research_report")
        }

app = FastAPI(title="Cross-Sell/Upsell Recommendation API")
agent = CrossSellUpsellAgent(DB_CONFIG)

@app.get("/recommendation")
async def get_recommendations(customer_id: str):
    try:
        result = agent.run_analysis(customer_id)
        return {
            "customer_id": customer_id,
            "research_report": result["research_report"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"API Error for customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)