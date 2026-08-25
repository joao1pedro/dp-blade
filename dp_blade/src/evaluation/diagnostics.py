import torch
import numpy as np
import pandas as pd
import os
import warnings

class DPDiagnosticLogger:
    def __init__(self, save_dir, experiment_name):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.grad_stats = []
        self.iakf_stats = []
        self.step_count = 0

    def _get_mean(self, val):
        """Função auxiliar para lidar com tensores e floats com segurança."""
        if isinstance(val, torch.Tensor):
            return torch.mean(val.float()).item()
        elif isinstance(val, (float, int)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return float(np.mean(val))
        return 0.0

    def log_step(self, model, optimizer):
        """Extrai as estatísticas do modelo e do estado do otimizador."""
        step_grad_data = {'step': self.step_count}
        step_iakf_data = {'step': self.step_count}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                l2_norm = torch.norm(grad, p=2).item()
                mean = torch.mean(grad).item()
                std = torch.std(grad).item()
                
                step_grad_data[f'{name}_l2'] = l2_norm
                step_grad_data[f'{name}_mean'] = mean
                step_grad_data[f'{name}_std'] = std
                step_grad_data[f'{name}_max'] = torch.max(grad).item()
                step_grad_data[f'{name}_min'] = torch.min(grad).item()
                
                if l2_norm > 100.0:
                    warnings.warn(f"Gradient Spike no passo {self.step_count} em {name}: L2={l2_norm:.2f}")
                if l2_norm < 1e-6:
                    warnings.warn(f"Gradient Collapse no passo {self.step_count} em {name}: L2={l2_norm:.2e}")
                
                state = optimizer.state.get(param, {})
                
                if 'innovation' in state:
                    step_iakf_data[f'{name}_innovation_l2'] = torch.norm(state['innovation'], p=2).item()
                
                if 'k_t' in state:
                    k_t_mean = self._get_mean(state['k_t'])
                    step_iakf_data[f'{name}_kt_mean'] = k_t_mean
                    if k_t_mean < 1e-4:
                        warnings.warn(f"Filtro Saturou (K_t -> 0) no passo {self.step_count} em {name}")
                        
                if 'p_t' in state:
                    step_iakf_data[f'{name}_pt_mean'] = self._get_mean(state['p_t'])
                    
                if 'q_t' in state:
                    step_iakf_data[f'{name}_qt_mean'] = self._get_mean(state['q_t'])
                    
        self.grad_stats.append(step_grad_data)
        self.iakf_stats.append(step_iakf_data)
        self.step_count += 1

    def save(self):
        """Salva os dados coletados em CSV."""
        if self.grad_stats:
            pd.DataFrame(self.grad_stats).to_csv(os.path.join(self.save_dir, f"{self.experiment_name}_grad_stats.csv"), index=False)
        if self.iakf_stats:
            pd.DataFrame(self.iakf_stats).to_csv(os.path.join(self.save_dir, f"{self.experiment_name}_iakf_stats.csv"), index=False)
        print(f"Diagnósticos salvos em {self.save_dir}")