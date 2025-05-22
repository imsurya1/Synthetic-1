
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

class AdvancedSynthesizer:
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.batch_size = 500
        self.epochs = 300
        self.hidden_dim = 256
        self.latent_dim = 128
        self.learning_rate = 1e-4
        
        # Quality parameters
        self.quality_threshold = 98
        self.min_value_threshold = 1e-10
        self.correlation_threshold = 0.98
        
        self.scalers = {}
        self.models = {}
        self.column_stats = {}
        
    def fit(self, data: pd.DataFrame, column_info: Dict[str, Any]):
        """Fit the synthesizer to the training data"""
        self.original_data = data.copy()
        self.column_info = column_info
        
        # Store column statistics
        for col in data.columns:
            self.column_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'skew': data[col].skew()
            }
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = scaler.fit_transform(data[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
            
        # Initialize models
        self.generator = Generator(input_dim=self.latent_dim, 
                                 hidden_dim=self.hidden_dim, 
                                 output_dim=len(data.columns)).to(self.device)
        self.discriminator = Discriminator(input_dim=len(data.columns), 
                                         hidden_dim=self.hidden_dim).to(self.device)
        
        # Train model
        self._train_gan(data)
        
    def _train_gan(self, data: pd.DataFrame):
        """Train GAN model"""
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        # Prepare training data
        train_data = torch.FloatTensor(np.concatenate([
            self.scalers[col].transform(data[col].values.reshape(-1, 1))
            for col in data.columns
        ], axis=1)).to(self.device)
        
        for epoch in range(self.epochs):
            # Train discriminator
            for _ in range(2):
                d_optimizer.zero_grad()
                
                # Real data
                real_output = self.discriminator(train_data)
                
                # Generate fake data
                noise = torch.randn(len(data), self.latent_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data.detach())
                
                # Calculate loss
                d_loss = -(torch.mean(torch.log(real_output + 1e-8)) + 
                          torch.mean(torch.log(1 - fake_output + 1e-8)))
                
                d_loss.backward()
                d_optimizer.step()
            
            # Train generator
            g_optimizer.zero_grad()
            fake_output = self.discriminator(fake_data)
            g_loss = -torch.mean(torch.log(fake_output + 1e-8))
            
            g_loss.backward()
            g_optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
                
    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data"""
        best_data = None
        best_quality = 0
        max_attempts = 10
        
        for attempt in range(max_attempts):
            # Generate synthetic data
            with torch.no_grad():
                noise = torch.randn(num_rows, self.latent_dim).to(self.device)
                synthetic_data = self.generator(noise).cpu().numpy()
            
            # Inverse transform data
            df_synthetic = pd.DataFrame()
            for idx, col in enumerate(self.original_data.columns):
                col_data = self.scalers[col].inverse_transform(synthetic_data[:, idx].reshape(-1, 1))
                
                # Ensure non-negative values where required
                if self.column_stats[col]['min'] >= 0:
                    col_data = np.maximum(col_data, self.min_value_threshold)
                    
                df_synthetic[col] = col_data.flatten()
            
            # Calculate quality score
            quality_score = self._calculate_quality(df_synthetic)
            
            if quality_score > best_quality:
                best_quality = quality_score
                best_data = df_synthetic.copy()
                
                if quality_score >= self.quality_threshold:
                    print(f"Achieved target quality score: {quality_score:.2f}")
                    return best_data
                    
        print(f"Best achieved quality score: {best_quality:.2f}")
        return best_data
        
    def _calculate_quality(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate quality score"""
        quality_score = 100.0
        
        for col in synthetic_data.columns:
            orig_data = self.original_data[col]
            synth_data = synthetic_data[col]
            
            # Distribution similarity
            ks_stat, _ = stats.ks_2samp(orig_data, synth_data)
            quality_score -= ks_stat * 10
            
            # Moment matching
            mean_diff = abs(orig_data.mean() - synth_data.mean()) / (abs(orig_data.mean()) + 1e-10)
            std_diff = abs(orig_data.std() - synth_data.std()) / (orig_data.std() + 1e-10)
            skew_diff = abs(orig_data.skew() - synth_data.skew()) / (abs(orig_data.skew()) + 1e-10)
            
            quality_score -= mean_diff * 5
            quality_score -= std_diff * 5
            quality_score -= skew_diff * 2
            
        # Correlation preservation
        orig_corr = self.original_data.corr()
        synth_corr = synthetic_data.corr()
        corr_diff = np.abs(orig_corr - synth_corr).mean().mean()
        quality_score -= corr_diff * 5
        
        return max(0, min(100, quality_score))

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
