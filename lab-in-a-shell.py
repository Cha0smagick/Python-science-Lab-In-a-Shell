import subprocess
from tqdm import tqdm

def install_dependencies(dependencies, category):
    print(f"\n{category} Dependencies:")
    print("======================")
    
    for dependency, description in dependencies.items():
        print(f"{dependency}: {description}")
    
    install_all = input(f"Do you want to install all {category} dependencies at once? (y/n): ").lower() == "y"
    
    if install_all:
        dependencies_list = list(dependencies.keys())
        print("Installing all dependencies...")
        subprocess.call(["pip", "install", *dependencies_list, "--quiet"])
        print("Installation completed.")
    else:
        while True:
            dependency = input("Enter the dependency you want to install (or type 'done' to finish): ")
            if dependency.lower() == "done":
                break
            elif dependency in dependencies:
                print(f"Installing {dependency}...")
                subprocess.call(["pip", "install", dependency, "--quiet"])
                print(f"{dependency} installed.")
            else:
                print("Invalid dependency. Please enter a valid dependency or type 'done' to finish.")

    print(f"\nInstallation of {category} dependencies completed.")

# Descriptions of Data Analysis Dependencies
data_analysis_dependencies = {
    "numpy": "Numerical computing library.",
    "pandas": "Data manipulation and analysis library.",
    "matplotlib": "Plotting library for creating visualizations.",
    "seaborn": "Data visualization library based on Matplotlib.",
    "scipy": "Scientific computing library.",
    "scikit-learn": "Machine learning library.",
    "statsmodels": "Statistical modeling library.",
    "plotly": "Interactive graphing library.",
    "bokeh": "Interactive visualization library.",
    "jupyter": "Interactive computing and data visualization.",
    "notebook": "Web-based interactive computational environment.",
    "ipython": "Interactive command-line interface.",
    "xlrd": "Library for reading Excel files.",
    "openpyxl": "Library for reading and writing Excel files.",
    "beautifulsoup4": "HTML and XML parsing library.",
    "requests": "HTTP library for making requests.",
    "pyyaml": "YAML parser and emitter.",
    "h5py": "HDF5 library for Python.",
    "sqlalchemy": "SQL toolkit and Object-Relational Mapping (ORM) library.",
    "psycopg2": "PostgreSQL adapter for Python.",
    "pymysql": "MySQL client library.",
    "pymongo": "Python driver for MongoDB.",
    "tqdm": "Progress bar library.",
    "joblib": "Library for lightweight pipelining in Python.",
    "numexpr": "Fast numerical expression evaluator.",
    "networkx": "Graph library.",
    "gensim": "Topic modeling and document similarity library.",
    "nltk": "Natural Language Toolkit.",
    "spacy": "Natural language processing library.",
    "fasttext": "Word embedding and text classification library.",
    "google-cloud-speech": "Google Cloud Speech-to-Text API client.",
    "ffmpeg-python": "FFmpeg wrapper.",
    "transformers": "State-of-the-art NLP library by Hugging Face.",
    "pandasai": "AI-driven automated data analysis and reporting.",
    # ... (Add descriptions for other dependencies) ...
}

# Descriptions of ML/DL Dependencies
ml_dl_dependencies = {
    "tensorflow": "Machine learning framework by Google.",
    "keras": "High-level neural networks API.",
    "scikit-learn": "Machine learning library.",
    "xgboost": "Gradient boosting library.",
    "lightgbm": "Gradient boosting framework.",
    "catboost": "Gradient boosting library with categorical features support.",
    "opencv-python": "Computer vision library.",
    "nltk": "Natural Language Toolkit.",
    "gensim": "Topic modeling and document similarity library.",
    "pymongo": "Python driver for MongoDB.",
    "networkx": "Graph library.",
    "torch": "Tensors and dynamic neural networks library.",
    "torchvision": "Image and video datasets library for PyTorch.",
    "transformers": "State-of-the-art NLP library by Hugging Face.",
    "spacy": "Natural language processing library.",
    "fasttext": "Word embedding and text classification library.",
    "pytorch-geometric": "Geometric deep learning extension for PyTorch.",
    "fastai": "Deep learning library built on PyTorch.",
    "graph-tool": "Graph theory library for Python.",
    "pybrain": "Modular Machine Learning Library for Python.",
    "theano": "Numerical computation library.",
    "lasagne": "Lightweight library to build and train neural networks in Theano.",
    "google-cloud-speech": "Google Cloud Speech-to-Text API client.",
    "ffmpeg-python": "FFmpeg wrapper.",
    "pandasai": "AI-driven automated data analysis and reporting.",
    "prophet": "Forecasting procedure library by Facebook.",
    "pycaret": "Low-code machine learning library.",
    "adversarial-robustness-toolbox": "Library for adversarial machine learning.",
    "alibi": "Algorithmic library for monitoring and explaining AI models.",
    "cleverhans": "Adversarial attacks and defenses library.",
    "pyro-ppl": "Probabilistic programming library.",
    "edward": "Probabilistic programming language.",
    "catalyst": "Deep learning library for PyTorch and Keras.",
    "kubeflow": "Machine learning toolkit for Kubernetes.",
    "rapids": "Accelerated data science library.",
    "tfx": "TensorFlow Extended for end-to-end ML pipelines.",
    "mlflow": "Open source platform for the complete machine learning lifecycle.",
    "pytorch-lightning": "Lightweight PyTorch wrapper for high-performance training.",
    "scikit-optimize": "Sequential model-based optimization library.",
    "hyperopt": "Distributed asynchronous hyperparameter optimization.",
    "optuna": "Hyperparameter optimization framework.",
    "ray": "Distributed computing library.",
    "shap": "Interpretable machine learning library.",
    "lime": "Local interpretable model-agnostic explanations library.",
    "tensorboard": "TensorFlow's visualization toolkit.",
    "neptune.ai": "ML Experiment Management.",
    "wandb": "Weights & Biases experiment tracking.",
    "visdom": "Interactive visualization library.",
    "comet-ml": "Experiment tracking and machine learning platform.",
    # ... (Add descriptions for other dependencies) ...
}

# Descriptions of Game Development Dependencies
game_dev_dependencies = {
    "pygame": "Cross-platform set of Python modules for video games.",
    "pyglet": "Cross-platform windowing and multimedia library.",
    "panda3d": "3D game engine.",
    "cocos2d": "2D game framework.",
    "arcade": "2D game library.",
    "pySFML": "Simple and Fast Multimedia Library.",
    "pyOpenGL": "OpenGL wrapper.",
    "godot-python": "Python bindings for the Godot game engine.",
    "kivent": "2D game engine specialized in performance.",
    "pymunk": "2D physics library.",
    "pybullet": "Physics engine for games, robotics, and simulations.",
    "ursina": "3D game engine for beginners.",
    "ppb": "Python programming environment for game development.",
    "pursuedpybear": "Experimental game engine.",
    "renpy": "Visual novel engine.",
    "pyxel": "Fantasy console for making retro games.",
    "pygame-zero": "Game framework for beginners.",
    "pygame-caster": "Game creation library for teachers.",
    "gym": "OpenAI toolkit for developing reinforcement learning algorithms.",
    "pySDL2": "Wrapper around the SDL2 library.",
    "google-cloud-speech": "Google Cloud Speech-to-Text API client.",
    "ffmpeg-python": "FFmpeg wrapper.",
    "pandasai": "AI-driven automated data analysis and reporting.",
    "godot": "2D and 3D game engine.",
    "cryengine": "Crytek's game engine.",
    "construct": "2D game engine for creating games without coding.",
    "defold": "Cross-platform game engine.",
    "gamemaker": "Game creation tool.",
    "libgdx": "Game development framework.",
    "cocos2d-x": "Game development framework.",
    "unity-python": "Python integration for Unity.",
    "inkscape": "Vector graphics editor.",
    "krita": "Digital painting and illustration software.",
    "pixlr": "Online photo editing tool.",
    "piskel": "Online sprite editor.",
    "autodesk-sketchbook": "Digital art application.",
    "spritely": "2D animation software.",
    "crazybump": "Texture map generation tool.",
    "substance-painter": "3D painting software.",
    "quixel-mixer": "Texture and material creation software.",
    "3d-coat": "3D sculpting software.",
    "zbrush": "Digital sculpting and painting software.",
    "meshmixer": "3D modeling software.",
    "makehuman": "Open-source character generator.",
    "make3d": "3D reconstruction software.",
    "synfig-studio": "2D animation software.",
    "animaker": "Online animation tool.",
    "plotagon": "3D animation software.",
    "toon-boom": "Animation software.",
    "dragonbones": "2D skeleton animation software.",
    "mixamo": "3D character animation service.",
    "spine": "2D skeletal animation software.",
    "brashmonkey-spriter": "2D sprite animation software.",
    "cocohub": "Coco/R-based compiler generator.",
    "raylib": "C library for game programming.",
    "aseprite": "Pixel art tool and animation software.",
    "blender": "Open-source 3D computer graphics software.",
    "unity-editor": "Game development environment.",
    "unreal": "Game engine by Epic Games.",
    "godot": "2D and 3D game engine.",
    "cryengine": "Crytek's game engine.",
    "construct": "2D game engine for creating games without coding.",
    "defold": "Cross-platform game engine.",
    "gamemaker": "Game creation tool.",
    "libgdx": "Game development framework.",
    "cocos2d-x": "Game development framework.",
    "pygame": "Cross-platform set of Python modules for video games.",
    "panda3d": "3D game engine.",
    "unity-python": "Python integration for Unity.",
    "renpy": "Visual novel engine.",
    "gdevelop": "Open-source game engine.",
    "twine": "Interactive fiction creation tool.",
    "ink": "Narrative scripting language for writing interactive stories.",
    "bfxr": "Sound effect generator.",
    "audacity": "Audio editing software.",
    "lmms": "Digital audio workstation.",
}

def install_python():
    subprocess.call(["python", "-m", "ensurepip", "--default-pip"])
    print("Python installation completed.")

def install_pip():
    subprocess.call(["python", "-m", "ensurepip", "--upgrade"])
    print("Pip installation completed.")

def main():
    print("Python Dependency Installer")
    print("==========================")
    print("This tool allows you to install various categories of Python dependencies.")
    
    while True:
        print("\nSelect an option:")
        print("1. Install Python")
        print("2. Install Pip")
        print("3. Install Data Analysis Dependencies")
        print("4. Install ML/DL Dependencies")
        print("5. Install Game Development Dependencies")
        print("6. Exit")
        
        choice = input("Enter your choice (1/2/3/4/5/6): ")
        
        if choice == "1":
            install_python()
        elif choice == "2":
            install_pip()
        elif choice == "3":
            install_dependencies(data_analysis_dependencies, "Data Analysis")
        elif choice == "4":
            install_dependencies(ml_dl_dependencies, "Machine Learning and Deep Learning")
        elif choice == "5":
            install_dependencies(game_dev_dependencies, "Game Development")
        elif choice == "6":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
