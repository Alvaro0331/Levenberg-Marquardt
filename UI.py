import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import LM as model
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.animation import FuncAnimation


# Creacion de la figura
def crear_figura():
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes([0.1, 0.1, 0.6, 0.8]) # [left, bottom, width, height]
    ax.set_title("Proyecto Final: Levenberg-Marquardt")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("X2", fontsize=12)
    ax.set_ylabel("X1", fontsize=12)
    #Dibujo de ejes
    ax.axhline(y=0, color='black', linewidth=0.7)
    ax.axvline(x=0, color='black', linewidth=0.7)
    
    #Leyenda
    leyenda = [
        plt.Line2D([0], [0], marker='o', color='w', label='Clase A', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Clase B', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], linestyle='-', color='green', label='Frontera'),
    ]
    ax.legend(handles=leyenda, loc='upper right', fontsize=8)
    return fig, ax


#Creacion de los widgets
def crear_widgets(fig):
    #Texto de clase
    class0Text=ax.text(0.75, 0.85, "Click izquierdo:", transform=fig.transFigure, fontsize=10, color='black')
    class0Text = ax.annotate(" Clase A",xycoords=(class0Text),xy=(1, 0), verticalalignment='bottom', fontsize=10, color='blue')
    class1Text=ax.text(0.75, 0.8, "Click derecho:", transform=fig.transFigure, fontsize=10, color='black')
    class1Text = ax.annotate(" Clase B",xycoords=(class1Text),xy=(1, 0), verticalalignment='bottom', fontsize=10, color='red')
    
    #TextBoxs
    LRBox=widgets.TextBox(plt.axes([0.85, 0.7, 0.1, 0.05]), 'Lambda (λ):', initial="0.01")
    NumNeuronsBox=widgets.TextBox(plt.axes([0.85, 0.6, 0.1, 0.05]), '# de neuronas:', initial="3")

    #RaddioButton para seleccionar dataset
    radioAx=plt.axes([0.75, 0.35, 0.1, 0.15])
    radioAx.set_title("Dataset", fontsize=10)
    radioDataset=widgets.RadioButtons(radioAx,('Custom', 'Option 1', 'Option 2'))

    # Botones
    plotButton=widgets.Button(plt.axes([0.75, 0.17, 0.1, 0.15]), 'Train', color='lightblue', hovercolor='skyblue')
    clearButton=widgets.Button(plt.axes([0.75, 0.10, 0.1, 0.05]), 'Clear', color='lightcoral', hovercolor='salmon')
    
    return plotButton, clearButton, radioDataset, NumNeuronsBox, LRBox


#Colocar puntos
puntos=[]
markers=[]
etiquetas=[]
lineas=[]
contorno_actual=[]
def onclick(event):
    if event.inaxes == ax:
        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            marker, = ax.plot(x, y, 'bo')  # Dibuja un punto azul en la posición clickeada
            puntos.append((x, y))  # Agrega el punto a la lista de puntos
            markers.append(marker)  # Agrega el objeto del punto a la lista de markers
            etiquetas.append(0)  # Agrega la etiqueta correspondiente a la lista de etiquetas
            fig.canvas.draw()  # Actualiza la figura para mostrar el nuevo punto
        elif event.button is MouseButton.RIGHT:
            x, y = event.xdata, event.ydata
            marker, = ax.plot(x, y, 'ro')  # Dibuja un punto rojo en la posición clickeada
            puntos.append((x, y))  # Agrega el punto a la lista de puntos
            markers.append(marker)  # Agrega el objeto del punto a la lista de markers
            etiquetas.append(1)  # Agrega la etiqueta correspondiente a la lista de etiquetas
            fig.canvas.draw()  # Actualiza la figura para mostrar el nuevo punto


#Limpiar puntos
def clear(event):
    for marker in markers:
        marker.remove()  # Elimina el punto de la figura
    for etiqueta in etiquetas:
        etiquetas.remove(etiqueta)  # Elimina la etiqueta de la lista de etiquetas
    markers.clear()  # Reinicia la lista de markers
    etiquetas.clear()
    puntos.clear() # Reinicia la lista de puntos
    clear_line()  # Limpia la línea de decisión


#Limpiar linea de decision
def clear_line():
    for c in contorno_actual:
        c.remove()
    contorno_actual.clear()
    for linea in lineas:
        if isinstance(linea, FuncAnimation):  # Si el contorno es una animación (FuncAnimation)
            if linea.event_source is not None:
                linea.event_source.stop()  # Detiene la animación
        else:
            linea.remove()  # Elimina el contorno de la figura
    lineas.clear()  # Reinicia la lista de contornos
    fig.canvas.draw()  # Actualiza la figura para reflejar los cambios


#Añadir puntos en base al dataset seleccionado
def dataset(label):
    clear(None) # Limpia los puntos y la línea de decisión antes de cargar el nuevo dataset
    rng = np.random.default_rng(42)
    if label == 'Option 1':
        # Dataset 1: XOR — Clase 0 en Q1 y Q3, Clase 1 en Q2 y Q4
        q1 = rng.uniform(1, 8, (20, 2))
        q3 = rng.uniform(-8, -1, (20, 2))
        class0 = np.vstack([q1, q3])
        q2 = np.column_stack([rng.uniform(-8, -1, 20), rng.uniform(1, 8, 20)])
        q4 = np.column_stack([rng.uniform(1, 8, 20), rng.uniform(-8, -1, 20)])
        class1 = np.vstack([q2, q4])
        puntos_dataset = [tuple(p) for p in np.vstack([class0, class1])]
        etiquetas_dataset = [0] * 40 + [1] * 40
    elif label == 'Option 2':
        # Dataset 2: Anillos — Clase 0 en círculo interior, Clase 1 en anillo exterior
        angles0 = rng.uniform(0, 2 * np.pi, 40)
        radii0 = rng.uniform(0.5, 3.5, 40)
        class0 = np.column_stack([radii0 * np.cos(angles0), radii0 * np.sin(angles0)])
        angles1 = rng.uniform(0, 2 * np.pi, 40)
        radii1 = rng.uniform(5.5, 9.0, 40)
        class1 = np.column_stack([radii1 * np.cos(angles1), radii1 * np.sin(angles1)])
        puntos_dataset = [tuple(p) for p in np.vstack([class0, class1])]
        etiquetas_dataset = [0] * 40 + [1] * 40
    else:
        # Custom dataset
        return
    # Agrega los puntos del dataset a la figura
    for (x,y), etiqueta in zip(puntos_dataset, etiquetas_dataset):
        color = 'bo' if etiqueta == 0 else 'ro'
        marker, = ax.plot(x, y, color)
        puntos.append((x, y))
        markers.append(marker)
        etiquetas.append(etiqueta)
    
    fig.canvas.draw()  # Actualiza la figura para mostrar los nuevos puntos


#Entrenar el modelo y dibujar la frontera de decisión
def train(event):
    clear_line()  # Limpia la línea de decisión antes de entrenar el modelo
    X=np.array(puntos, dtype=np.float64)
    d=np.array(etiquetas, dtype=np.float64).reshape(-1, 1)
    LR=float(LRBox.text)
    NumNeurons=int(NumNeuronsBox.text)
    global contorno_actual
    contorno_actual=[]

    if len(X) < 2:
        return
    
    mlp = model.MLP([2, NumNeurons, 1])
    mlp.train(X, d, max_iter=1000, lam=LR, verbose=False)

    clear_line()

    # Meshgrid para la frontera de decisión
    xs = np.linspace(-10, 10, 200)
    ys = np.linspace(-10, 10, 200)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]

    def update(frame):
        epoch_data = mlp.history[frame]
        mlp.W1 = epoch_data['W1']
        mlp.b1 = epoch_data['b1']
        mlp.W2 = epoch_data['W2']
        mlp.b2 = epoch_data['b2']

        #Eliminar contorno anterior
        if contorno_actual:
            contorno_actual[0].remove()  # contourf
            contorno_actual[1].remove()  # contour
            contorno_actual.clear()
        
        #Dibujar regiones y frontera
        Z = mlp.predict(grid).reshape(xx.shape)
        cf = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#AACCFF', '#FFAAAA'], alpha=0.4)
        c = ax.contour(xx, yy, Z, levels=[0.5], colors='green')
        contorno_actual.append(cf)
        contorno_actual.append(c)

        ax.set_title(f"Epoch: {epoch_data['epoch']}, Loss: {epoch_data['loss']:.4f}")

    anim = FuncAnimation(fig, update, frames=len(mlp.history), interval=50,repeat=False)
    lineas.append(anim)
    fig.canvas.draw()


fig, ax = crear_figura()
plotButton, clearButton, radioDataset, NumNeuronsBox, LRBox = crear_widgets(fig)
fig.canvas.mpl_connect('button_press_event', onclick)
clearButton.on_clicked(clear)
radioDataset.on_clicked(dataset)
plotButton.on_clicked(train)
plt.show()