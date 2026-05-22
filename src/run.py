"""
Entry point for the path tracer.

Prompts the user to either render a new image or view/compare saved renders.
"""


import src.renderer as renderer
import src.viewer as viewer


while True:
    choice = input("Type \"render\" to rerender the scene or \"compare\" to compare saved images.\n").strip().lower()

    if choice == "render":
        renderer.start()
        viewer.view()
        break
    elif choice == "compare":
        viewer.view()
        break
    else:
        print("Invalid choice. Try again.")
