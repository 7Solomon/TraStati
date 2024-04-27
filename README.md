# TraStati

Generierung und erkennung statischer Systeme mittels Transformer

# Usage

### Generierung eines Datensets

```python
python main --data
```

<img src="assets/cut_image.jpg" width="200" height="200">
<img src="assets/rotated_inage.jpg" width="200" height="200">
<img src="assets/noised_image.jpg" width="200" height="200">

### Visualisierung eines Datasets

```python
python main --display
```

### Visualisierung des outputs eines Models

```python
python main --test
```

### Trainieren eines Models

```python
python main --train
```

# Installation

```python
pip install -r requirements.txt
```

Da die Daten mit "pdflatex" mit python subprocess generiert werden, müssen pdflatex, convert installiert sein.

## Other dependencies

### Linux (Debian)

```console
$ sudo apt install texlive
$ sudo apt install imagemagick

```
