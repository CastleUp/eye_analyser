import pandas as pd
import matplotlib.pyplot as plt

def generate_plot():
    # Load data
    try:
        df = pd.read_csv('comparison_log.csv')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot distances
    plt.plot(df['Timestamp'], df['V1_Distance'], label='V1: ArcFace (General Face Model)', color='blue', linewidth=2)
    plt.plot(df['Timestamp'], df['V2_Distance'], label='V2: DINOv2 (Vision Transformer)', color='orange', linewidth=1.5, alpha=0.8)
    
    # Add threshold line (default 0.3)
    plt.axhline(y=0.28, color='red', linestyle='--', label='Threshold (0.28)')
    
    plt.title('Comparison of Eye Recognition Models (Cosine Distance)', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Distance (lower is more similar)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    plt.savefig('recognition_comparison.png', dpi=300)
    print("Plot saved as 'recognition_comparison.png'")
    # plt.show()

if __name__ == "__main__":
    generate_plot()
