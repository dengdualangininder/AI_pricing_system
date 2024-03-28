# AI Pricing System

This is an AI pricing system that enables users to generate suggested retail prices by simply uploading a CSV file formatted accordingly. The system utilizes an AI model trained through ensemble learning, specifically stacking, which is detailed in the accompanying Jupyter Notebook (`retail_price.ipynb`). This notebook explains the interpretation of the training data (`retail_price.csv`) and the process of training the model.

## Files Included:
- `retail_price.ipynb`: Jupyter Notebook detailing the interpretation of training data and the model training process.
- `app.py`: Python file containing the Streamlit frontend interface, allowing users to upload data and receive suggested retail prices.
- `retail_price.csv`: Sample CSV file for training data interpretation and model training.

## Instructions for Use:
1. Ensure you have the necessary Python packages installed by running `pip install -r requirements.txt`.
2. Open and run `retail_price.ipynb` in a Jupyter Notebook environment to understand the training data and model training process.
3. Execute `app.py` to launch the Streamlit frontend interface.
4. Upload a CSV file with the appropriate format to receive suggested retail prices.
5. The system utilizes Gemini AI to provide explanations for the pricing decisions.

## Note:
- The system's accuracy and effectiveness may vary based on the quality and relevance of the input data.
- Feel free to explore and customize the AI model and interface according to your specific requirements and preferences.
