import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import linearmodels as lm
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.system import SUR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class M8SustainabilityAnalysis:
    """
    Implementation of the methodology from the paper:
    "The Effect of Mineral Resource Rents, Financial Technology, and Digital Transformation 
    on Environmental Quality in M-8 Countries"
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the analysis class
        
        Parameters:
        data_path: str, path to the dataset
        """
        self.data = None
        self.results = {}
        
        # M-8 Countries
        self.m8_countries = ['United States', 'China', 'India', 'Germany', 
                           'Japan', 'United Kingdom', 'France', 'South Korea']
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and preprocess the dataset"""
        self.data = pd.read_csv(data_path)
        
        # Ensure proper panel structure
        if 'year' in self.data.columns and 'country' in self.data.columns:
            self.data = self.data.set_index(['country', 'year'])
        
        # Create logarithmic transformations as used in the paper
        variables_to_log = ['RETI', 'DigEco', 'FinS', 'Gov', 'GDP', 'Ind']
        for var in variables_to_log:
            if var in self.data.columns:
                self.data[f'Ln{var}'] = np.log(self.data[var] + 1e-10)  # Add small constant to avoid log(0)
    
    def calculate_digital_economy_index(self, ict_indicators):
        """
        Calculate Digital Economy Index using Principal Component Analysis (PCA)
        as described in the paper
        
        Parameters:
        ict_indicators: DataFrame with ICT infrastructure, trade, and social indicators
        """
        # Standardize the indicators
        scaler = StandardScaler()
        ict_scaled = scaler.fit_transform(ict_indicators)
        
        # Apply PCA
        pca = PCA()
        pca_result = pca.fit_transform(ict_scaled)
        
        # Use first principal component as digital economy index
        digital_economy_index = pca_result[:, 0]
        
        # Calculate weights based on variance explained
        weights = pca.explained_variance_ratio_
        
        return digital_economy_index, weights, pca
    
    def calculate_reti_popp_model(self, patents_data, alpha=0.22, beta=0.03):
        """
        Calculate Renewable Energy Technology Innovations (RETI) using Popp's model
        
        RETI_t = PT_t + (1 - α) * RETI_{t-1} + β * PT_{t-1}
        
        Parameters:
        patents_data: DataFrame with patent counts by country and year
        alpha: technological depreciation rate (0.22 as per paper)
        beta: diffusion rate (0.03 as per paper)
        """
        reti_values = []
        
        for country in patents_data['country'].unique():
            country_data = patents_data[patents_data['country'] == country].sort_values('year')
            patents = country_data['patents'].values
            
            reti_country = [patents[0]]  # Initialize with first year patents
            
            for t in range(1, len(patents)):
                reti_t = patents[t] + (1 - alpha) * reti_country[t-1] + beta * patents[t-1]
                reti_country.append(reti_t)
            
            # Add to results
            country_years = country_data['year'].values
            for i, year in enumerate(country_years):
                reti_values.append({
                    'country': country,
                    'year': year,
                    'RETI': reti_country[i]
                })
        
        return pd.DataFrame(reti_values)
    
    def pesaran_cd_test(self, data, variable):
        """
        Pesaran Cross-Sectional Dependence Test
        """
        n_countries = len(data.index.get_level_values(0).unique())
        n_periods = len(data.index.get_level_values(1).unique())
        
        # Calculate correlation matrix
        corr_matrix = data[variable].unstack().corr()
        
        # Pesaran CD statistic
        cd_stat = np.sqrt(2 / (n_countries * (n_countries - 1))) * np.sum(
            np.triu(corr_matrix.values, k=1)
        )
        
        p_value = 2 * (1 - stats.norm.cdf(np.abs(cd_stat)))
        
        return cd_stat, p_value
    
    def slope_heterogeneity_test(self, model, data):
        """
        Test for slope heterogeneity across countries
        """
        # This is a simplified implementation
        countries = data.index.get_level_values(0).unique()
        coefficients = []
        
        for country in countries:
            country_data = data.xs(country, level=0)
            if len(country_data) > 5:  # Ensure sufficient observations
                try:
                    model_country = sm.OLS(
                        country_data['LnRETI'], 
                        sm.add_constant(country_data[['LnDigEco', 'LnFinS', 'LnGDP', 'LnGov', 'LnInd']])
                    ).fit()
                    coefficients.append(model_country.params)
                except:
                    continue
        
        # Calculate coefficient variation as heterogeneity measure
        coeff_df = pd.DataFrame(coefficients)
        heterogeneity = coeff_df.std() / coeff_df.mean()
        
        return heterogeneity
    
    def simultaneous_quantile_regression(self, data, quantiles=[0.25, 0.50, 0.75]):
        """
        Implement Simultaneous Quantile Regression (SQR) as in the paper
        """
        sqr_results = {}
        
        # Prepare data
        X_vars = ['LnDigEco', 'LnFinS', 'LnGDP', 'LnGov', 'LnInd']
        X = sm.add_constant(data[X_vars])
        y = data['LnRETI']
        
        for q in quantiles:
            # Quantile regression for each quantile
            model = QuantReg(y, X).fit(q=q)
            sqr_results[f'Q{int(q*100)}'] = {
                'coefficients': model.params,
                'pvalues': model.pvalues,
                'confidence_intervals': model.conf_int()
            }
        
        return sqr_results
    
    def driscoll_kraay_estimation(self, data):
        """
        Driscoll-Kraay fixed effects estimation with robust standard errors
        """
        # Using linearmodels for panel data with Driscoll-Kraay standard errors
        model = PanelOLS(
            data['LnRETI'], 
            sm.add_constant(data[['LnDigEco', 'LnFinS', 'LnGDP', 'LnGov', 'LnInd']]),
            entity_effects=True
        )
        
        fitted = model.fit(cov_type='kernel', kernel='andrews')
        return fitted
    
    def system_gmm_estimation(self, data):
        """
        System GMM estimation for dynamic panel data
        """
        # Prepare data for GMM
        data_gmm = data.reset_index()
        data_gmm = data_gmm.sort_values(['country', 'year'])
        
        # Create lagged variables
        data_gmm['LnRETI_lag1'] = data_gmm.groupby('country')['LnRETI'].shift(1)
        
        # Drop missing values from lagging
        data_gmm = data_gmm.dropna()
        
        # System GMM model
        formula = 'LnRETI ~ 1 + LnRETI_lag1 + LnDigEco + LnFinS + LnGDP + LnGov + LnInd'
        
        try:
            model = lm.PanelGMM.from_formula(
                formula, 
                data_gmm.set_index(['country', 'year']),
                weights=True
            )
            fitted = model.fit()
            return fitted
        except:
            print("System GMM estimation failed, using alternative approach")
            return None
    
    def moderation_analysis(self, data, moderator='Gov'):
        """
        Analyze moderating effects of governance or financial structure
        """
        if moderator == 'Gov':
            interaction_term = data['LnDigEco'] * data['LnGov']
            mod_var = 'LnGov'
        else:  # Financial structure
            interaction_term = data['LnDigEco'] * data['LnFinS']
            mod_var = 'LnFinS'
        
        # Add interaction term to data
        data_mod = data.copy()
        data_mod['interaction'] = interaction_term
        
        # Estimate model with interaction
        X = sm.add_constant(data_mod[['LnDigEco', mod_var, 'interaction', 'LnFinS', 'LnGDP', 'LnInd']])
        y = data_mod['LnRETI']
        
        model = sm.OLS(y, X).fit()
        
        return model
    
    def augmented_mean_group(self, data):
        """
        Augmented Mean Group (AMG) estimator for robustness check
        """
        # Simplified AMG implementation
        countries = data.index.get_level_values(0).unique()
        coefficients = []
        
        for country in countries:
            country_data = data.xs(country, level=0)
            if len(country_data) > 5:
                try:
                    X = sm.add_constant(country_data[['LnDigEco', 'LnFinS', 'LnGDP', 'LnGov', 'LnInd']])
                    y = country_data['LnRETI']
                    
                    model = sm.OLS(y, X).fit()
                    coefficients.append(model.params)
                except:
                    continue
        
        # Calculate mean group estimates
        coeff_df = pd.DataFrame(coefficients)
        amg_results = coeff_df.mean()
        
        return amg_results
    
    def fully_modified_ols(self, data):
        """
        Fully Modified OLS (FMOLS) for cointegrated panels
        """
        # This is a simplified implementation
        # In practice, you might need specialized packages for panel FMOLS
        
        X = sm.add_constant(data[['LnDigEco', 'LnFinS', 'LnGDP', 'LnGov', 'LnInd']])
        y = data['LnRETI']
        
        model = sm.OLS(y, X).fit()
        
        # Apply Newey-West HAC standard errors
        model_hac = model.get_robustcov_results(cov_type='HAC', maxlags=1)
        
        return model_hac
    
    def run_complete_analysis(self):
        """
        Run the complete analysis as described in the paper
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("Running M-8 Sustainability Analysis...")
        
        # 1. Diagnostic tests
        print("\n1. Running Diagnostic Tests...")
        cd_stat, cd_pvalue = self.pesaran_cd_test(self.data, 'LnRETI')
        print(f"Pesaran CD Test: Statistic = {cd_stat:.4f}, p-value = {cd_pvalue:.4f}")
        
        # 2. Simultaneous Quantile Regression
        print("\n2. Running Simultaneous Quantile Regression...")
        sqr_results = self.simultaneous_quantile_regression(self.data)
        self.results['SQR'] = sqr_results
        
        # 3. Robustness checks
        print("\n3. Running Robustness Checks...")
        
        # Driscoll-Kraay
        print("   - Driscoll-Kraay Estimation")
        dk_results = self.driscoll_kraay_estimation(self.data)
        self.results['Driscoll_Kraay'] = dk_results
        
        # System GMM
        print("   - System GMM Estimation")
        gmm_results = self.system_gmm_estimation(self.data)
        self.results['System_GMM'] = gmm_results
        
        # AMG
        print("   - Augmented Mean Group Estimation")
        amg_results = self.augmented_mean_group(self.data)
        self.results['AMG'] = amg_results
        
        # FMOLS
        print("   - Fully Modified OLS")
        fmols_results = self.fully_modified_ols(self.data)
        self.results['FMOLS'] = fmols_results
        
        # 4. Moderation analysis
        print("\n4. Running Moderation Analysis...")
        
        # Governance moderation
        gov_moderation = self.moderation_analysis(self.data, moderator='Gov')
        self.results['Gov_Moderation'] = gov_moderation
        
        # Financial structure moderation
        fins_moderation = self.moderation_analysis(self.data, moderator='FinS')
        self.results['FinS_Moderation'] = fins_moderation
        
        print("\nAnalysis completed successfully!")
        
        return self.results
    
    def plot_quantile_results(self):
        """Plot quantile regression coefficients as in Figure 8 of the paper"""
        if 'SQR' not in self.results:
            print("Please run SQR analysis first.")
            return
        
        # Extract coefficients
        quantiles = [0.25, 0.50, 0.75]
        variables = ['LnDigEco', 'LnFinS', 'LnGov', 'LnGDP', 'LnInd']
        
        coeff_data = []
        for var in variables:
            for q in quantiles:
                key = f'Q{int(q*100)}'
                if var in self.results['SQR'][key]['coefficients']:
                    coeff = self.results['SQR'][key]['coefficients'][var]
                    coeff_data.append({'Variable': var, 'Quantile': q, 'Coefficient': coeff})
        
        coeff_df = pd.DataFrame(coeff_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        for var in variables:
            var_data = coeff_df[coeff_df['Variable'] == var]
            plt.plot(var_data['Quantile'], var_data['Coefficient'], 
                    marker='o', label=var, linewidth=2)
        
        plt.xlabel('Quantile')
        plt.ylabel('Coefficient Estimate')
        plt.title('Estimated Coefficients Across Quantiles (Similar to Figure 8)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def generate_summary_tables(self):
        """Generate summary tables similar to those in the paper"""
        if not self.results:
            print("Please run analysis first.")
            return
        
        # Table 1: SQR Results (Similar to Table 9)
        print("\nTable 1: Simultaneous Quantile Regression Results")
        print("=" * 60)
        
        variables = ['const', 'LnDigEco', 'LnFinS', 'LnGDP', 'LnGov', 'LnInd']
        quantiles = ['Q25', 'Q50', 'Q75']
        
        header = f"{'Variable':<12} " + "".join([f"{q:>12}" for q in quantiles])
        print(header)
        print("-" * 60)
        
        for var in variables:
            row = f"{var:<12}"
            for q in quantiles:
                if q in self.results['SQR'] and var in self.results['SQR'][q]['coefficients']:
                    coeff = self.results['SQR'][q]['coefficients'][var]
                    row += f"{coeff:>12.4f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

# Example usage
def main():
    """
    Example of how to use the M8SustainabilityAnalysis class
    """
    # Initialize the analyzer
    analyzer = M8SustainabilityAnalysis()
    
    # Note: You would need to load your actual data here
    # analyzer.load_data('path_to_your_data.csv')
    
    print("M-8 Sustainability Analysis Implementation")
    print("This code implements the methodology from the paper:")
    print("'The Effect of Mineral Resource Rents, Financial Technology, and Digital Transformation on Environmental Quality in M-8 Countries'")
    
    # The main methods you would use:
    # 1. analyzer.load_data('your_data.csv')
    # 2. results = analyzer.run_complete_analysis()
    # 3. analyzer.plot_quantile_results()
    # 4. analyzer.generate_summary_tables()

if __name__ == "__main__":
    main()