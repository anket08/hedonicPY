import sys
import logging
import re
import warnings
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np
import os
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Scikit-Learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional dependencies for Research-Grade Boosting/Explainability
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed. Fallback to RandomForest for residuals.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed.")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    logger.warning("CatBoost not installed.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Explainability features will be skipped.")


# Vectorized Haversine distance
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees) - Vectorized for Pandas
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def tiered_shipping_cost(distance):
    """
    Tiered pricing model for transport cost estimation.
    """
    return np.where(distance <= 200, 3000, 
             np.where(distance <= 800, 5000 + (distance - 200) * 4,
                      8000 + (distance - 800) * 3))

def calculate_rto_vectorized(prices, powers, inter_state=True):
    """
    Calculate RTO slab based on price and cc/power and inter_state transfer
    """
    # Base rates: small < 150cc, med 150-300cc, high > 300cc
    rates = np.where(powers <= 150, 0.04, 
                 np.where(powers <= 300, 0.06, 0.08))
    
    # If inter state transfer, add extra 2% tax penalty for NOC and Re-Reg
    rates = np.where(inter_state, rates + 0.02, rates)
        
    rto = prices * rates
    return np.maximum(rto, 3500) # Floor threshold

class BikeArbitrageEngine:
    def __init__(self, bike_data_path, geo_data_path):
        self.bike_data_path = bike_data_path
        self.geo_data_path = geo_data_path
        self.df = None
        self.geo_df = None
        self.hedonic_model = None
        self.residual_model = None
        self.market_clusters = None
        self.brand_premiums = {}
        
    def run_pipeline(self):
        """Execute full end-to-end engine pipeline."""
        logger.info("Starting Bike Arbitrage Engine Pipeline...")
        self.load_data()
        self.engineer_features()
        self.detect_outliers()
        self.train_hedonic_model()
        self.train_residual_model()
        self.cluster_markets()
        arbitrage_df = self.find_arbitrage_opportunities()
        self.run_explainability()
        self.save_artifacts()
        return arbitrage_df
        
    def load_data(self):
        logger.info(f"Loading data from {self.bike_data_path} & {self.geo_data_path}")
        self.df = pd.read_csv(self.bike_data_path)
        
        # Clean Bikes
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        # Apply sanity bounds for realistic modeling
        self.df = self.df[
            (self.df['price'] > 5000) & (self.df['price'] < 2000000) &
            (self.df['age'] > 0) & (self.df['age'] <= 30) &
            (self.df['kms_driven'] > 100) & (self.df['power'] > 50)
        ].reset_index(drop=True)
        
        # Clean Geo Data
        self.geo_df = pd.read_csv(self.geo_data_path)[['Location', 'Latitude', 'Longitude']]
        self.geo_df['Location'] = self.geo_df['Location'].astype(str).str.split("Latitude").str[0]
        self.geo_df.columns = ['city', 'latitude', 'longitude']
        
        # Normalize city names
        normalize = lambda name: re.sub(r"\(.*?\)", "", str(name)).strip().title()
        self.df['city'] = self.df['city'].apply(normalize)
        self.geo_df['city'] = self.geo_df['city'].apply(normalize)
        
        # Merge Geo Early to filter unplottable cities
        self.df = self.df.drop(columns=['latitude', 'longitude'], errors='ignore')
        self.df = self.df.merge(self.geo_df.drop_duplicates(subset=['city']), on='city', how='left')
        self.df = self.df.dropna(subset=['latitude', 'longitude', 'price']).reset_index(drop=True)
        
        logger.info(f"Data Loaded: {self.df.shape[0]} valid bike records ready.")

    def engineer_features(self):
        logger.info("Engineering advanced features...")
        
        # Log Transformations
        self.df['log_price'] = np.log1p(self.df['price'])
        self.df['log_kms'] = np.log1p(self.df['kms_driven'])
        self.df['log_power'] = np.log1p(self.df['power'])
        
        # Basic Categorical mappings
        self.df['first_owner'] = (self.df['owner'] == "First Owner").astype(int)
        
        # Ensure brand column exists or extract it from bike_name
        if 'brand' not in self.df.columns:
            self.df['brand'] = self.df['bike_name'].str.split().str[0]
            
        premium_brands = ['KTM', 'Triumph', 'Ducati', 'BMW', 'Harley-Davidson', 'Kawasaki']
        self.df['is_premium_brand'] = self.df['brand'].isin(premium_brands).astype(int)
        
        # Depreciation & Usage signals
        self.df['kms_per_year'] = self.df['kms_driven'] / self.df['age']
        self.df['log_kms_per_year'] = np.log1p(self.df['kms_per_year'])
        self.df['age_squared'] = self.df['age'] ** 2
        self.df['depreciation_proxy'] = self.df['power'] / self.df['age']
        self.df['log_price_per_cc'] = np.log1p(self.df['price'] / self.df['power'])
        
        # Interactions
        self.df['power_age_interaction'] = self.df['log_power'] * self.df['age']
        self.df['owner_age_interaction'] = (1 - self.df['first_owner']) * self.df['age']

        # Brand mean target encoding (with smoothing)
        brand_means = self.df.groupby('brand')['log_price'].agg(['mean', 'count'])
        global_mean = self.df['log_price'].mean()
        smoothing = 10
        self.brand_premiums = (brand_means['count'] * brand_means['mean'] + smoothing * global_mean) / (brand_means['count'] + smoothing)
        self.df['brand_premium_encoded'] = self.df['brand'].map(self.brand_premiums)

        # Market & Liquidity Signals
        model_counts = self.df['bike_name'].value_counts()
        self.df['model_frequency'] = self.df['bike_name'].map(model_counts)
        
        city_supply = self.df['city'].value_counts()
        self.df['city_supply_count'] = self.df['city'].map(city_supply)
        
        # Liquidity Score (Log of sample size scaled 1-10)
        self.df['liquidity_score'] = np.clip(np.log1p(self.df['model_frequency']) / np.log1p(model_counts.max()) * 10, 1, 10)
        
        logger.info("Feature engineering complete.")

    def detect_outliers(self):
        logger.info("Running Isolation Forest for anomaly detection...")
        # Use isolation forest on core numeric metrics to drop bad data points
        iso_features = ['log_price', 'age', 'log_kms', 'log_power']
        iso_model = IsolationForest(contamination=0.03, random_state=42)
        outliers = iso_model.fit_predict(self.df[iso_features].fillna(0))
        self.df['is_outlier'] = (outliers == -1)
        
        # Drop extreme anomalies
        dropped = self.df['is_outlier'].sum()
        self.df = self.df[~self.df['is_outlier']].reset_index(drop=True)
        logger.info(f"Dropped {dropped} outlier records.")

    def train_hedonic_model(self):
        logger.info("Training ElasticNetCV Hedonic Model...")
        self.hedonic_features = [
            'age', 'age_squared', 'log_kms', 'log_power', 'first_owner',
            'kms_per_year', 'depreciation_proxy', 'power_age_interaction',
            'owner_age_interaction', 'brand_premium_encoded', 'is_premium_brand'
        ]
        
        # Ensure 'claimed_mileage_detailed' exists and is numeric before appending
        if 'claimed_mileage_detailed' in self.df.columns:
            self.df['claimed_mileage_detailed'] = pd.to_numeric(self.df['claimed_mileage_detailed'], errors='coerce').fillna(self.df['claimed_mileage_detailed'].median())
            self.hedonic_features.append('claimed_mileage_detailed')
        
        X_hed = self.df[self.hedonic_features].copy()
        # Fill strictly needed simple NAs
        X_hed = X_hed.fillna(X_hed.median())
        y_hed = self.df['log_price']
        
        # Setup Pipeline with scaling
        self.hedonic_model = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], random_state=42))
        ])
        
        self.hedonic_model.fit(X_hed, y_hed)
        
        # Predictions
        self.df['log_hedonic_price'] = self.hedonic_model.predict(X_hed)
        self.df['hedonic_price'] = np.expm1(self.df['log_hedonic_price'])
        self.df['log_residual'] = self.df['log_price'] - self.df['log_hedonic_price']
        
        r2 = r2_score(y_hed, self.df['log_hedonic_price'])
        logger.info(f"Hedonic Model R2 Base: {r2:.4f}")

    def train_residual_model(self):
        logger.info("Training Residual Ensemble Models...")
        self.residual_features = self.hedonic_features + [
            'city_supply_count', 'liquidity_score', 'model_frequency'
        ]
        
        # Add city target encoding to capture city premiums smoothly
        city_means = self.df.groupby('city')['log_residual'].mean()
        self.df['city_residual_premium'] = self.df['city'].map(city_means).fillna(0)
        self.residual_features.append('city_residual_premium')

        X_res = self.df[self.residual_features].fillna(0)
        y_res = self.df['log_residual']
        
        # Split Data purely to observe metrics locally 
        # (Production uses full data fit at the end)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        
        models = []
        if HAS_XGB:
            xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            models.append(('xgb', xgb_model))
        if HAS_LGB:
            lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, 
                                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
            models.append(('lgb', lgb_model))
        if HAS_CB:
            cb_model = cb.CatBoostRegressor(n_estimators=300, depth=6, learning_rate=0.05, 
                                            random_seed=42, verbose=0, thread_count=-1)
            models.append(('cb', cb_model))
            
        if not models: # Fallback
            rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            models.append(('rf', rf_model))

        logger.info(f"Using models for tracking residuals: {[name for name, _ in models]}")

        # Train Ensemble
        if len(models) > 1:
            self.residual_model = VotingRegressor(estimators=models)
        else:
            self.residual_model = models[0][1]

        # Evaluate just for logs
        self.residual_model.fit(X_train, y_train)
        preds = self.residual_model.predict(X_test)
        
        # Final combined assessment
        log_price_true = y_test + self.df.loc[X_test.index, 'log_hedonic_price']
        log_price_pred = preds + self.df.loc[X_test.index, 'log_hedonic_price']
        
        combo_r2 = r2_score(log_price_true, log_price_pred)
        logger.info(f"Combined (Hedonic + Residual) Out-Of-Sample R2: {combo_r2:.4f}")

        # Fit on full dataset for Engine use
        self.residual_model.fit(X_res, y_res)
        self.df['residual_pred'] = self.residual_model.predict(X_res)
        self.df['log_market_price'] = self.df['log_hedonic_price'] + self.df['residual_pred']
        self.df['market_adjusted_price'] = np.expm1(self.df['log_market_price'])

    def cluster_markets(self):
        logger.info("Clustering cities into demand/supply zones...")
        # Get city aggregates
        city_stats = self.df.groupby('city').agg({
            'log_residual': 'mean', # High residual = Overpriced = High Demand/Low Supply
            'price': 'median',
            'model_frequency': 'sum',
            'latitude': 'first',
            'longitude': 'first'
        }).fillna(0)
        
        scaler = StandardScaler()
        cluster_data = scaler.fit_transform(city_stats[['log_residual', 'price', 'model_frequency']])
        
        # K-Means with 4 zones
        kmeans = KMeans(n_clusters=4, random_state=42)
        city_stats['demand_zone'] = kmeans.fit_predict(cluster_data)
        
        # Map back to main frame
        self.df['demand_zone'] = self.df['city'].map(city_stats['demand_zone'])

    def run_explainability(self):
        """Generates optional SHAP values if module installed"""
        if not HAS_SHAP:
            return
            
        logger.info("Generating SHAP explainability insights...")
        try:
            # We attempt SHAP on the first tree model in the VotingRegressor or the standalone model
            if isinstance(self.residual_model, VotingRegressor):
                explainer_model = self.residual_model.estimators_[0]
            else:
                explainer_model = self.residual_model
                
            X_res = self.df[self.residual_features].fillna(0)
            explainer = shap.TreeExplainer(explainer_model)
            # Sample for speed
            shap_values = explainer.shap_values(X_res.sample(n=min(1000, len(X_res)), random_state=42))
            logger.info("Successfully generated SHAP values.")
        except Exception as e:
            logger.warning(f"Failed to generate SHAP values: {e}")

    def find_arbitrage_opportunities(self):
        logger.info("Scanning for cross-city arbitrage opportunities comprehensively...")
        
        # Segment by Model and Age bucket
        self.df['age_bucket'] = pd.cut(self.df['age'], bins=[0,3,6,10,20, 50], labels=['0-3','3-6','6-10','10-20','20+'])
        
        # Aggregate logic
        model_city_stats = self.df.groupby(['bike_name', 'age_bucket', 'city']).agg({
            'price': ['mean', 'std', 'count'],
            'market_adjusted_price': 'mean',
            'latitude': 'first',
            'longitude': 'first',
            'power': 'mean',
            'liquidity_score': 'mean',
            'demand_zone': 'first'
        }).reset_index()
        
        model_city_stats.columns = ['bike_name', 'age_bucket', 'city', 'avg_buy_price', 'std_price', 'inventory_count', 
                                    'avg_market_price', 'lat', 'lon', 'power', 'liquidity', 'zone']
        
        # Filter down to entries that actually exist and have liquidity
        model_city_stats = model_city_stats[model_city_stats['inventory_count'] > 0]
        model_city_stats['std_price'] = model_city_stats['std_price'].fillna(model_city_stats['avg_buy_price'] * 0.1)
        
        results = []
        
        # Cross join mechanism
        unique_models = model_city_stats['bike_name'].unique()
        logger.info(f"Processing combinatorial cross-joins for {len(unique_models)} unique bike models...")
        
        for name in unique_models:
            subset = model_city_stats[model_city_stats['bike_name'] == name]
            for bucket in subset['age_bucket'].unique():
                bucket_subset = subset[subset['age_bucket'] == bucket]
                
                if len(bucket_subset) < 2:
                    continue
                    
                buy_df = bucket_subset.copy().rename(columns={
                    'city': 'buy_city', 'avg_buy_price': 'buy_price', 'std_price': 'buy_std',
                    'lat': 'buy_lat', 'lon': 'buy_lon', 'power': 'buy_power', 
                    'inventory_count': 'buy_inventory', 'zone':'buy_zone'
                })
                
                sell_df = bucket_subset.copy().rename(columns={
                    'city': 'sell_city', 'avg_market_price': 'sell_price', 'std_price': 'sell_std',
                    'lat': 'sell_lat', 'lon': 'sell_lon', 'liquidity': 'sell_liquidity',
                    'inventory_count': 'sell_inventory', 'zone':'sell_zone'
                })
                
                # Cross match combinations
                merged = pd.merge(buy_df, sell_df, on=['bike_name', 'age_bucket'], how='inner')
                merged = merged[merged['buy_city'] != merged['sell_city']]
                
                if merged.empty:
                    continue
                    
                # Vectorized Cost & Returns logic
                merged['distance_km'] = haversine_vectorized(
                    merged['buy_lat'], merged['buy_lon'], merged['sell_lat'], merged['sell_lon']
                )
                
                merged['transport_cost'] = tiered_shipping_cost(merged['distance_km'])
                
                # Are they in the same state? Rough heuristic using distance (or external state mapping if provided)
                merged['is_interstate'] = merged['distance_km'] > 150 
                merged['rto_cost'] = calculate_rto_vectorized(merged['buy_price'], merged['buy_power'], merged['is_interstate'])
                
                # Negotiation margin = roughly 5% discount obtained physically
                negotiated_purchase = merged['buy_price'] * 0.95 
                
                merged['total_cost'] = negotiated_purchase + merged['transport_cost'] + merged['rto_cost'] + 2000 # 2000 base overhead
                merged['gross_profit'] = merged['sell_price'] - merged['buy_price']
                merged['net_profit'] = merged['sell_price'] - merged['total_cost']
                
                # Only keep mathematically profitable paths
                merged = merged[merged['net_profit'] > 1000]
                
                if not merged.empty:
                    # Risk Scoring
                    # Distance penalty (higher distance = higher risk), Liquidity penalty (inverse), Price volatility penalty
                    merged['risk_score'] = (
                        (merged['distance_km'] / 1000) * 0.2 + 
                        (10 / merged['sell_liquidity']) * 0.4 + 
                        (merged['sell_std'] / merged['sell_price']) * 0.4
                    )
                    
                    # Confidence scoring: Logistic function of sample size and low std deviations
                    sample_mass = merged['buy_inventory'] + merged['sell_inventory']
                    merged['confidence'] = 1 / (1 + np.exp(-0.5 * (sample_mass - 5))) # S-curve over sample size
                    
                    # Final risk-weighted output
                    merged['risk_adjusted_profit'] = (merged['net_profit'] * merged['confidence']) / (1 + merged['risk_score'])
                    merged['est_days_to_sell'] = np.clip(100 / merged['sell_liquidity'], 5, 90).astype(int)
                    
                    results.append(merged)
                    
        if not results:
            logger.warning("No profitable arbitrage opportunities found after filtering.")
            return pd.DataFrame()
            
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.sort_values('risk_adjusted_profit', ascending=False).reset_index(drop=True)
        
        # Clean Final Output Columns
        output_cols = [
            'bike_name', 'age_bucket', 'buy_city', 'sell_city', 'distance_km',
            'buy_price', 'sell_price', 'total_cost', 'net_profit', 
            'risk_score', 'confidence', 'risk_adjusted_profit', 
            'sell_liquidity', 'est_days_to_sell', 'buy_zone', 'sell_zone'
        ]
        
        logger.info(f"Engine completed! Found {len(final_df)} arbitrage pathways.")
        return final_df[output_cols].head(500) # Only return top 500 for dashboards

    def save_artifacts(self):
        logger.info("Saving ML models and market matrices to models/artifacts.pkl...")
        os.makedirs("models", exist_ok=True)
        
        # Save reference data
        geo_dict = self.geo_df.drop_duplicates('city').set_index('city')[['latitude', 'longitude']].to_dict('index') if self.geo_df is not None else {}
        demand_zones = self.df.drop_duplicates('city').set_index('city')['demand_zone'].to_dict() if self.df is not None else {}
        city_residuals = self.df.drop_duplicates('city').set_index('city')['city_residual_premium'].to_dict() if self.df is not None else {}
        model_liquidity = self.df.drop_duplicates('bike_name').set_index('bike_name')['liquidity_score'].to_dict() if self.df is not None else {}
        
        artifacts = {
            'hedonic_model': self.hedonic_model,
            'residual_model': self.residual_model,
            'brand_premiums': self.brand_premiums,
            'geo_dict': geo_dict,
            'demand_zones': demand_zones,
            'city_residuals': city_residuals,
            'model_liquidity': model_liquidity,
            'hedonic_features': getattr(self, 'hedonic_features', []),
            'residual_features': getattr(self, 'residual_features', [])
        }
        joblib.dump(artifacts, "models/artifacts.pkl")
        logger.info("Artifacts saved successfully.")

    @classmethod
    def evaluate_new_entry(cls, bike_dict):
        """
        Instantly evaluates a new bike entry against all known destination markets 
        using saved ML artifacts. Assumes run_pipeline() has been executed once.
        """
        if not os.path.exists("models/artifacts.pkl"):
            raise FileNotFoundError("ML artifacts not found. Please run the main engine pipeline once.")
            
        artifacts = joblib.load("models/artifacts.pkl")
        
        # Create a single row dataframe
        df = pd.DataFrame([bike_dict])
        
        # Feature Engineering Pipeline for a single row
        df['log_price'] = np.log1p(df['buy_price'])
        df['log_kms'] = np.log1p(df['kms_driven'])
        df['log_power'] = np.log1p(df['power'])
        
        # Clean brand and map properly
        df['brand'] = df['bike_name'].str.split().str[0]
        premium_brands = ['KTM', 'Triumph', 'Ducati', 'BMW', 'Harley-Davidson', 'Kawasaki']
        df['is_premium_brand'] = df['brand'].isin(premium_brands).astype(int)
        
        df['kms_per_year'] = df['kms_driven'] / df['age']
        df['log_kms_per_year'] = np.log1p(df['kms_per_year'])
        df['age_squared'] = df['age'] ** 2
        df['depreciation_proxy'] = df['power'] / df['age']
        df['log_price_per_cc'] = np.log1p(df['buy_price'] / df['power'])
        
        df['power_age_interaction'] = df['log_power'] * df['age']
        df['owner_age_interaction'] = (1 - df['first_owner']) * df['age']
        
        df['brand_premium_encoded'] = df['brand'].map(artifacts['brand_premiums']).fillna(
            sum(artifacts['brand_premiums'].values()) / len(artifacts['brand_premiums'])
        )
        
        df['liquidity_score'] = df['bike_name'].map(artifacts['model_liquidity']).fillna(1.0)
        
        # We don't have frequency counts for a single new entry, so we use robust defaults 
        # based on liquidity to prevent errors during inference
        df['city_supply_count'] = 10 
        df['model_frequency'] = 10
        df['claimed_mileage_detailed'] = df.get('claimed_mileage', 40)
        
        # Setup features for Hedonic Pricing
        X_hed = df[[f for f in artifacts['hedonic_features'] if f in df.columns]]
        # Assign missing columns with 0
        for col in artifacts['hedonic_features']:
            if col not in X_hed.columns: X_hed[col] = 0
            
        X_hed = X_hed[artifacts['hedonic_features']].copy()
        
        hedonic_model = artifacts['hedonic_model']
        log_hedonic = hedonic_model.predict(X_hed)[0]
        
        # Now, duplicate this row for EVERY known city in India to test sales destinations
        cities = list(artifacts['geo_dict'].keys())
        eval_df = pd.DataFrame([df.iloc[0].to_dict()] * len(cities))
        eval_df['sell_city'] = cities
        
        # Apply city-specific residual premiums
        eval_df['city_residual_premium'] = eval_df['sell_city'].map(artifacts['city_residuals']).fillna(0)
        
        # Prepare residual features
        X_res = eval_df[[f for f in artifacts['residual_features'] if f in eval_df.columns]].copy()
        for col in artifacts['residual_features']:
            if col not in X_res.columns: X_res[col] = 0
            
        X_res = X_res[artifacts['residual_features']].copy()
        
        residual_model = artifacts['residual_model']
        residual_preds = residual_model.predict(X_res)
        
        # Final pricing array
        eval_df['log_market_price'] = log_hedonic + residual_preds
        eval_df['sell_price'] = np.expm1(eval_df['log_market_price'])
        
        buy_city = df['buy_city'].iloc[0]
        if buy_city in artifacts['geo_dict']:
            buy_lat = artifacts['geo_dict'][buy_city]['latitude']
            buy_lon = artifacts['geo_dict'][buy_city]['longitude']
        else:
            return {"error": f"Buy city {buy_city} not found in intelligence database."}
            
        # Add dest lat/lon
        eval_df['sell_lat'] = eval_df['sell_city'].apply(lambda x: artifacts['geo_dict'][x]['latitude'])
        eval_df['sell_lon'] = eval_df['sell_city'].apply(lambda x: artifacts['geo_dict'][x]['longitude'])
        
        # Calculate distances & costs manually for this array
        eval_df['distance_km'] = haversine_vectorized(
            buy_lat, buy_lon, eval_df['sell_lat'], eval_df['sell_lon']
        )
        
        eval_df['transport_cost'] = tiered_shipping_cost(eval_df['distance_km'])
        eval_df['is_interstate'] = eval_df['distance_km'] > 150
        eval_df['rto_cost'] = calculate_rto_vectorized(eval_df['buy_price'], eval_df['power'], eval_df['is_interstate'])
        
        eval_df['total_cost'] = eval_df['buy_price'] * 0.95 + eval_df['transport_cost'] + eval_df['rto_cost'] + 2000
        eval_df['net_profit'] = eval_df['sell_price'] - eval_df['total_cost']
        
        # Filter same city
        eval_df = eval_df[eval_df['sell_city'] != buy_city]
        eval_df = eval_df[eval_df['net_profit'] > 0]
        
        eval_df = eval_df.sort_values('net_profit', ascending=False)
        
        if eval_df.empty:
            return {"status": "no_paths", "message": "No profitable destinations found for this entry."}
            
        top_destinations = eval_df.head(3)
        results = []
        for _, row in top_destinations.iterrows():
            results.append({
                "sell_city": row['sell_city'],
                "sell_price": int(row['sell_price']),
                "net_profit": int(row['net_profit']),
                "distance_km": int(row['distance_km']),
                "transport_cost": int(row['transport_cost']),
                "rto_cost": int(row['rto_cost'])
            })
            
        return {
            "status": "success",
            "bike_name": bike_dict['bike_name'],
            "buy_city": buy_city,
            "buy_price": int(bike_dict['buy_price']),
            "base_hedonic_value": int(np.expm1(log_hedonic)),
            "destinations": results
        }

if __name__ == "__main__":
    BIKE_CSV = "Used_Bikes_mileage_KTMboost.csv"
    GEO_CSV = "Indian Cities Geo Data.csv"
    
    engine = BikeArbitrageEngine(bike_data_path=BIKE_CSV, geo_data_path=GEO_CSV)
    
    # Run pipeline
    results_df = engine.run_pipeline()
    
    if not results_df.empty:
        print("\n--- TOP 10 RISK-ADJUSTED ARBITRAGE OPPORTUNITIES ---")
        print(results_df.head(10).to_string(index=False))
        
        # Save output for Dashboard
        results_df.to_csv("Arbitrage_Opportunities_Output.csv", index=False)
        print("\nResults successfully exported to: Arbitrage_Opportunities_Output.csv")
    else:
        print("\nNo viable arbitrage routes met the risk requirements.")
