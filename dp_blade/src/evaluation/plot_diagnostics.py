import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (14, 6)})

def load_data(exp_dir, exp_name):
    """Carrega os CSVs de gradientes e do IAKF."""
    grad_path = f"{exp_dir}/{exp_name}_grad_stats.csv"
    iakf_path = f"{exp_dir}/{exp_name}_iakf_stats.csv"
    
    try:
        df_grad = pd.read_csv(grad_path)
        df_iakf = pd.read_csv(iakf_path)
        return df_grad, df_iakf
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None

def get_mean_across_layers(df, suffix):
    """Calcula a média de uma métrica através de todas as camadas em cada passo."""
    cols = [c for c in df.columns if c.endswith(suffix)]
    if not cols:
        return np.zeros(len(df))
    return df[cols].mean(axis=1)

def apply_smoothing(series, window=50):
    """Aplica média móvel para reduzir o ruído visual no gráfico."""
    return series.rolling(window=window, min_periods=1).mean()

def plot_kalman_gain_comparison(df_nlp, df_vision, window=50):
    """Compara o Ganho de Kalman (K_t) entre NLP e Visão. 
       Se K_t cai para zero rápido, o filtro saturou."""
    
    kt_nlp = get_mean_across_layers(df_nlp, '_kt_mean')
    kt_vision = get_mean_across_layers(df_vision, '_kt_mean')
    
    plt.figure()
    plt.plot(df_nlp['step'], apply_smoothing(kt_nlp, window), label='NLP (Sucesso)', color='blue', linewidth=2)
    plt.plot(df_vision['step'], apply_smoothing(kt_vision, window), label='Visão (Falha)', color='red', linewidth=2)
    
    plt.title("Evolução do Ganho de Kalman (K_t) Médio - NLP vs Visão")
    plt.xlabel("Passos de Treinamento (Steps)")
    plt.ylabel("K_t (Média entre camadas)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_innovation_vs_gradient(df_grad, df_iakf, modality_name, window=50):
    """Plota a Norma L2 do Gradiente original versus a Norma L2 da Inovação.
       Se a inovação é muito maior que o gradiente, o ruído DP engoliu o sinal."""
    
    grad_l2 = get_mean_across_layers(df_grad, '_l2')
    innovation_l2 = get_mean_across_layers(df_iakf, '_innovation_l2')
    
    plt.figure()
    plt.plot(df_grad['step'], apply_smoothing(grad_l2, window), label='Norma L2 do Grad. Bruto', color='green', alpha=0.7)
    plt.plot(df_iakf['step'], apply_smoothing(innovation_l2, window), label='Norma L2 da Inovação (nu_t)', color='purple', alpha=0.7)
    
    plt.title(f"Gradiente vs Inovação IAKF ({modality_name})")
    plt.xlabel("Passos de Treinamento (Steps)")
    plt.ylabel("Norma L2")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_gradient_stability(df_grad_nlp, df_grad_vision, window=50):
    """Compara a estabilidade da Norma L2 dos gradientes (NLP vs Visão)."""
    
    l2_nlp = get_mean_across_layers(df_grad_nlp, '_l2')
    l2_vision = get_mean_across_layers(df_grad_vision, '_l2')
    
    plt.figure()
    plt.plot(df_grad_nlp['step'], apply_smoothing(l2_nlp, window), label='NLP', color='blue')
    plt.plot(df_grad_vision['step'], apply_smoothing(l2_vision, window), label='Visão', color='red')
    
    plt.title("Estabilidade do Gradiente (Norma L2 Média) - Escala Log")
    plt.xlabel("Passos de Treinamento (Steps)")
    plt.ylabel("Norma L2 (Média entre camadas)")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    VISION_DIR = "./saved_models/cifar10_ablation/diagnostics"
    NLP_DIR = "./saved_models/qnli_ablation/diagnostics"
    
    VISION_EXP = "image_cifar10_IAKF"
    NLP_EXP = "text_qnli_IAKF"

    print("Carregando dados...")
    df_grad_vis, df_iakf_vis = load_data(VISION_DIR, VISION_EXP)
    df_grad_nlp, df_iakf_nlp = load_data(NLP_DIR, NLP_EXP)

    if df_grad_vis is not None and df_grad_nlp is not None:
        print("Gerando gráficos...")
        
        plot_kalman_gain_comparison(df_iakf_nlp, df_iakf_vis, window=20)
        
        plot_gradient_stability(df_grad_nlp, df_grad_vis, window=20)
        
        plot_innovation_vs_gradient(df_grad_vis, df_iakf_vis, "Visão", window=20)
        plot_innovation_vs_gradient(df_grad_nlp, df_iakf_nlp, "NLP", window=20)