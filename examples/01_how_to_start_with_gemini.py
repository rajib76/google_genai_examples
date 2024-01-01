# Author : Rajib Deb
# Date : Jan 1 2024
# Description: This program shows an example of how to use gemini pro for text generation
# to run this examples, please do a "pip install -r requirements.txt" first
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Get your api key from https://makersuite.google.com/
google_api_key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=google_api_key)
logger = logging.getLogger("__name__")


class GeminiGenAI():
    def __init__(self, model_name="gemini-pro"):
        logger.log(level=logging.INFO, msg="Instantiating the model")
        self.module = "GeminiGenAI"
        self.model_name = model_name

    def get_response(self, prompt: str, stream=False):
        """
        This function calls the gemini model with the prompt to get a response back
        from the model
        :param prompt: The prompt from the used
        :param stream: Whether to stream the repsonse
        :return: Returns the response
        """
        try:
            logger.log(level=logging.INFO, msg="Sending prompt to the language model")
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt, stream=stream)
            logger.log(level=logging.INFO, msg="Obtained response from the language model")
            return response
        except Exception as e:
            logger.log(level=logging.ERROR, msg="Failed to get response")
            print(e)

    def get_response_feedback(self, response):
        """
        This module will retrun the safety ratings of the response THis is a beautiful feature
        which shows the probability of harmfulness in a reponse.
        :param response: The response from the LLM
        :return: Returns the safety rating of the response
        """
        return response.prompt_feedback


if __name__ == "__main__":
    gemini_llm = GeminiGenAI()
    stream = True
    response = gemini_llm.get_response("What is Artifical Intelligence", stream=stream)
    print("LLM response...\n")

    if stream:
        # If stream is true, we need to get each chunk and print it
        # retruning the response and trying to print it will cause an error
        for chunk in response:
            print(chunk.text)
    else:
        print(response.text)

    # Getting the feedback from the response
    response_feedback = gemini_llm.get_response_feedback(response)
    print("LLM response feedback...\n")
    print(response_feedback)
