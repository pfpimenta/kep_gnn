# Check for cuda capabilities
if command -v nvidia-smi; then
    if nvidia-smi | grep 'CUDA Version: 11.[345]'; then
        install_type='cudatoolkit=11.3'
    elif nvidia-smi | grep 'CUDA Version: 11.[6789]'; then
        install_type='cudatoolkit=11.6'
    elif nvidia-smi | grep 'CUDA Version: 1[23456789].'; then
        install_type='cudatoolkit=11.6'
    elif nvidia-smi | grep 'CUDA Version: 11.[012]'; then
        install_type='cudatoolkit=10.3'
    elif nvidia-smi | grep 'CUDA Version: 10.[23456789]'; then
        install_type='cudatoolkit=10.3'
    else
        install_type='cpuonly'
    fi
else
    install_type='cpuonly'
fi

conda create --force -n venv_kep_gnn

if command -v mamba; then
    if echo ${install_type} | grep 'cpuonly'; then
        # Mamba fails to solve cpuonly pyg https://github.com/mamba-org/mamba/issues/1542
        install_method='conda'
    else
        install_method='mamba'
    fi
else
    install_method='conda'
fi

command ${install_method} install -n venv_kep_gnn -c pytorch -c pyg -c conda-forge pyg pytorch=1.11 torchvision=0.12 torchaudio=0.11 ${install_type} networkx matplotlib ipykernel pre-commit pandas

echo "done"
