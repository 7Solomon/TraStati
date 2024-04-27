# TraStati

Generierung und erkennung statischer Systeme mittels Transformer

# Usage

### Generierung eines Datensets

```python
python main --data
```

<div>
    <div style="height: 300px; width: 300px; border: 1px solid black;">
        <img src="assets/cut_image.jpg" height="200" alt="Cut Image">
        <p style="word-wrap: break-word;">Cut the Image - Als erstes wird das Image auf eine fest größe Zugeschnitten</p>
    </div>
    <div style="height: 300px; width: 300px; border: 1px solid black;">
        <img src="assets/rotated_image.jpg" height="200" alt="Rotated Image">
        <p style="word-wrap: break-word;">Rotate the Image - Danach wird das Image random Rotiert</p>
    </div>
    <div style="height: 300px; width: 300px; border: 1px solid black;">
        <img src="assets/noised_image.jpg" height="200" alt="Noised Image">
        <p style="word-wrap: break-word;">Noise the Image - Zum schluss wird das Image per Trapez Form randomized und die weißen Pixel werden zu einer Papier ähnlichen Farbe genoised</p>
    </div>
    <div style="height: 300px; width: 300px; border: 1px solid black;">
        <img src="assets/output_image.jpg" height="200" alt="Output Image">
        <p style="word-wrap: break-word;">Label of the Image - Hier sieht man die ground Truths der Images</p>
    </div>
</div>

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
