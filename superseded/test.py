import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def on_key(event):
    print(f"you pressed {event.key}")
    if event.key == 'p':
        print("P key was pressed")

fig.canvas.mpl_connect('key_press_event', on_key)

ax.plot([1, 2, 3], [1, 2, 3])
plt.show()
