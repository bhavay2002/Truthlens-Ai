"""
Exploratory Data Analysis (EDA) for Fake News Dataset
Provides comprehensive data analysis and visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from collections import Counter
import re
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class FakeNewsEDA:
    """Comprehensive EDA for fake news detection dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {}
    
    def basic_info(self):
        """Display basic dataset information"""
        logger.info("=" * 60)
        logger.info("BASIC DATASET INFORMATION")
        logger.info("=" * 60)
        
        logger.info(f"Dataset Shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        logger.info(f"\nData Types:\n{self.df.dtypes}")
        logger.info(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"\nMissing Values:\n{missing[missing > 0]}")
        else:
            logger.info("\n✓ No missing values")
        
        # Duplicates
        dups = self.df.duplicated().sum()
        logger.info(f"\nDuplicate Rows: {dups} ({dups/len(self.df)*100:.2f}%)")
        
        self.report['basic_info'] = {
            'shape': self.df.shape,
            'duplicates': dups,
            'missing': missing.to_dict()
        }
    
    def label_distribution(self, save_path='reports/figures/label_distribution.png'):
        """Analyze and visualize label distribution"""
        logger.info("\n" + "=" * 60)
        logger.info("LABEL DISTRIBUTION")
        logger.info("=" * 60)
        
        if 'label' not in self.df.columns:
            logger.warning("No 'label' column found")
            return
        
        label_counts = self.df['label'].value_counts()
        label_pcts = self.df['label'].value_counts(normalize=True) * 100
        
        logger.info(f"\nLabel Counts:")
        logger.info(f"  Real (0): {label_counts.get(0, 0):,} ({label_pcts.get(0, 0):.2f}%)")
        logger.info(f"  Fake (1): {label_counts.get(1, 0):,} ({label_pcts.get(1, 0):.2f}%)")
        
        # Check for imbalance
        imbalance_ratio = label_counts.max() / label_counts.min()
        if imbalance_ratio > 1.5:
            logger.warning(f"⚠ Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        else:
            logger.info(f"✓ Classes are balanced (ratio: {imbalance_ratio:.2f}:1)")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sns.barplot(x=['Real', 'Fake'], y=label_counts.values, palette=['green', 'red'], ax=axes[0])
        axes[0].set_title('Label Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(label_counts.values):
            axes[0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(label_counts.values, labels=['Real', 'Fake'], autopct='%1.1f%%',
                    colors=['green', 'red'], startangle=90)
        axes[1].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to: {save_path}")
        plt.close()
        
        self.report['label_distribution'] = label_counts.to_dict()
    
    def text_length_analysis(self, save_path='reports/figures/text_length.png'):
        """Analyze text length statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("TEXT LENGTH ANALYSIS")
        logger.info("=" * 60)
        
        if 'text' not in self.df.columns:
            logger.warning("No 'text' column found")
            return
        
        self.df['text_length'] = self.df['text'].astype(str).str.len()
        self.df['word_count'] = self.df['text'].astype(str).str.split().str.len()
        
        # Overall statistics
        logger.info("\nOverall Text Statistics:")
        logger.info(f"  Avg Length: {self.df['text_length'].mean():.0f} characters")
        logger.info(f"  Avg Words: {self.df['word_count'].mean():.0f} words")
        logger.info(f"  Min Length: {self.df['text_length'].min()}")
        logger.info(f"  Max Length: {self.df['text_length'].max()}")
        
        # By label
        if 'label' in self.df.columns:
            logger.info("\nBy Label:")
            for label in sorted(self.df['label'].unique()):
                label_name = 'Real' if label == 0 else 'Fake'
                subset = self.df[self.df['label'] == label]
                logger.info(f"  {label_name}:")
                logger.info(f"    Avg Length: {subset['text_length'].mean():.0f} chars")
                logger.info(f"    Avg Words: {subset['word_count'].mean():.0f} words")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Character length distribution
        axes[0, 0].hist(self.df['text_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Text Length (Characters)', fontweight='bold')
        axes[0, 0].set_xlabel('Character Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['text_length'].mean(), color='red', linestyle='--', 
                          label=f"Mean: {self.df['text_length'].mean():.0f}")
        axes[0, 0].legend()
        
        # Word count distribution
        axes[0, 1].hist(self.df['word_count'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_title('Distribution of Word Count', fontweight='bold')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(self.df['word_count'].mean(), color='red', linestyle='--',
                          label=f"Mean: {self.df['word_count'].mean():.0f}")
        axes[0, 1].legend()
        
        # Box plot by label (if available)
        if 'label' in self.df.columns:
            self.df['label_name'] = self.df['label'].map({0: 'Real', 1: 'Fake'})
            sns.boxplot(x='label_name', y='text_length', data=self.df, ax=axes[1, 0])
            axes[1, 0].set_title('Text Length by Label', fontweight='bold')
            axes[1, 0].set_xlabel('Label')
            axes[1, 0].set_ylabel('Character Count')
            
            sns.boxplot(x='label_name', y='word_count', data=self.df, ax=axes[1, 1])
            axes[1, 1].set_title('Word Count by Label', fontweight='bold')
            axes[1, 1].set_xlabel('Label')
            axes[1, 1].set_ylabel('Word Count')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to: {save_path}")
        plt.close()
    
    def word_frequency_analysis(self, top_n=20, save_path='reports/figures/word_frequency.png'):
        """Analyze most common words"""
        logger.info("\n" + "=" * 60)
        logger.info("WORD FREQUENCY ANALYSIS")
        logger.info("=" * 60)
        
        if 'text' not in self.df.columns:
            return
        
        # Common stop words to remove
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                         'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                         'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                         'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their'])
        
        def get_words(text):
            words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
            return [w for w in words if w not in stop_words]
        
        all_words = []
        for text in self.df['text']:
            all_words.extend(get_words(text))
        
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(top_n)
        
        logger.info(f"\nTop {top_n} Most Common Words:")
        for word, count in most_common[:10]:
            logger.info(f"  {word}: {count:,}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        words, counts = zip(*most_common)
        sns.barplot(x=list(counts), y=list(words), palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Most Common Words', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Word')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved to: {save_path}")
        plt.close()
    
    def generate_wordclouds(self, save_path='reports/figures/wordclouds.png'):
        """Generate word clouds for fake and real news"""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING WORD CLOUDS")
        logger.info("=" * 60)
        
        if 'text' not in self.df.columns or 'label' not in self.df.columns:
            logger.warning("Need both 'text' and 'label' columns")
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Real news wordcloud
            real_text = ' '.join(self.df[self.df['label'] == 0]['text'].astype(str))
            wordcloud_real = WordCloud(width=800, height=400, background_color='white',
                                      colormap='Greens', max_words=100).generate(real_text)
            axes[0].imshow(wordcloud_real, interpolation='bilinear')
            axes[0].set_title('Real News - Word Cloud', fontsize=16, fontweight='bold')
            axes[0].axis('off')
            
            # Fake news wordcloud
            fake_text = ' '.join(self.df[self.df['label'] == 1]['text'].astype(str))
            wordcloud_fake = WordCloud(width=800, height=400, background_color='white',
                                      colormap='Reds', max_words=100).generate(fake_text)
            axes[1].imshow(wordcloud_fake, interpolation='bilinear')
            axes[1].set_title('Fake News - Word Cloud', fontsize=16, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved to: {save_path}")
            plt.close()
        except ImportError:
            logger.warning("WordCloud not installed. Install with: pip install wordcloud")
    
    def detect_data_quality_issues(self):
        """Detect potential data quality problems"""
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY CHECKS")
        logger.info("=" * 60)
        
        issues = []
        
        if 'text' in self.df.columns:
            # Very short texts
            short_texts = (self.df['text'].astype(str).str.len() < 50).sum()
            if short_texts > 0:
                pct = short_texts / len(self.df) * 100
                issues.append(f"Very short texts (< 50 chars): {short_texts} ({pct:.1f}%)")
            
            # Very long texts
            long_texts = (self.df['text'].astype(str).str.len() > 10000).sum()
            if long_texts > 0:
                pct = long_texts / len(self.df) * 100
                issues.append(f"Very long texts (> 10,000 chars): {long_texts} ({pct:.1f}%)")
            
            # Empty or whitespace only
            empty_texts = self.df['text'].astype(str).str.strip().str.len().eq(0).sum()
            if empty_texts > 0:
                issues.append(f"Empty/whitespace texts: {empty_texts}")
        
        # Duplicates
        dups = self.df.duplicated().sum()
        if dups > 0:
            pct = dups / len(self.df) * 100
            issues.append(f"Duplicate rows: {dups} ({pct:.1f}%)")
        
        if issues:
            logger.warning("⚠ Data Quality Issues Found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ No major data quality issues detected")
        
        self.report['quality_issues'] = issues
    
    def generate_full_report(self, output_dir='reports/figures'):
        """Run all EDA analyses"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING FULL EDA REPORT")
        logger.info("=" * 70 + "\n")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.basic_info()
        self.label_distribution(f'{output_dir}/label_distribution.png')
        self.text_length_analysis(f'{output_dir}/text_length.png')
        self.word_frequency_analysis(save_path=f'{output_dir}/word_frequency.png')
        self.generate_wordclouds(f'{output_dir}/wordclouds.png')
        self.detect_data_quality_issues()
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ EDA REPORT COMPLETE!")
        logger.info(f"✓ Figures saved to: {output_dir}/")
        logger.info("=" * 70 + "\n")
        
        return self.report


def run_eda_from_csv(csv_path: str, output_dir='reports/figures'):
    """Convenience function to run EDA on a CSV file"""
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    eda = FakeNewsEDA(df)
    report = eda.generate_full_report(output_dir)
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        run_eda_from_csv(csv_path)
    else:
        logger.info("Usage: python eda.py <path_to_csv>")
        logger.info("Example: python src/data/eda.py data/raw/fake.csv")
