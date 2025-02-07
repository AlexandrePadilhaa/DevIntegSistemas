
### Ambiente virtual (venv)

python -m venv venv

venv\Scripts\activate

python main.py

deactivate

## Entradas para o programa cgne e gcnr
 
python 'signal_path' 'signal_shape_x' 'signal_shape_y' 'matriz_path' 'matriz_shape_x' 'matriz_shape_y' 'resutl_path' 'result_shape' 'signal_name'
python cgne.py "signal\A-60x60-1.csv" 50816 1 "data\H-1.csv" 50816 3600 "result/" 60 "cgne-A"

## Entradas para o programa gcnr
python 'signal_path' 'signal_shape_x' 'signal_shape_y' 'matriz_path' 'matriz_shape_x' 'matriz_shape_y' 'resutl_path' 'result_shape' 'signal_name'
python cgnr.py "signal\A-30x30-1.csv" 27904 1 "data\H-2.csv" 27904 900 "result/" 30 "GCRN-A-1"
python cgnr.py "signal\g-30x30-1.csv" 27904 1 "data\H-2.csv" 27904 900 "result/" 30 "GCRN-G-1"
python cgnr.py "signal\g-30x30-2.csv" 27904 1 "data\H-2.csv" 27904 900 "result/" 30 "GCRN-G-2"


## Observações

Baixar matriz H 


