# Failure Analysis

We evaluated the End-to-End LLM Generation using the LLM-as-a-judge (which scored an average of 2.99 / 5). Below are 5 distinct failure modes observed in the pipeline:

### 1. Generic Apology Loop
**Description**: The model frequently defaults to a generic apology and requests the user to DM their order details, completely ignoring specific context or questions asked by the customer.
**Example**:
- *Customer*: "My tracking says delivered to the mail room but the mail room doesn't have it."
- *Generated Reply*: "I'm sorry to hear that. Please DM us your order details so we can assist you further."

### 2. Hallucinated Policies and Links
**Description**: The model invents tracking links, carrier names, or Amazon policies that do not exist, attempting to be "helpful" but providing misleading information.
**Example**:
- *Customer*: "Where is my package?"
- *Generated Reply*: "We're sorry! Please check your tracking at www.amazon.com/fake-tracking-link to see the updated carrier info."

### 3. Retrieval Distraction
**Description**: When the retrieval index pulls a past interaction that is only tangentially related (e.g. they both mention the word "Prime"), the model copies the retrieved response entirely, answering the wrong question.
**Example**:
- *Customer*: "Prime delivery is late again."
- *Generated Reply*: "We're sorry you're having trouble logging into Prime Video. Please reset your password here." (Copied from a retrieved Prime Video thread).

### 4. Incorrect Escalation / Lack of Empathy
**Description**: The model fails to recognize high-frustration or critical issues (like stolen items) and responds with a robotic, cheerful tone instead of escalating to a human.
**Example**:
- *Customer*: "THIS IS THE THIRD TIME MY PACKAGE WAS STOLEN I AM FURIOUS!"
- *Generated Reply*: "Oh no! :D We're here to help! Please check your porch again."

### 5. Multilingual Failure
**Description**: The model sometimes receives Spanish or French tweets, but replies in English, or attempts to translate it poorly, frustrating the customer.
**Example**:
- *Customer*: "¿Dónde está mi paquete?"
- *Generated Reply*: "I'm sorry, I don't understand. Where is your package? Please DM us."
