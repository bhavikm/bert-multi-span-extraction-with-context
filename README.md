# bert-multi-span-extraction-with-context
BERT model variant to extract multiple spans from text with context tokens.

BERT has been adopted for Question Answering tasks which involves extracting a single-span of tokens from text with a given question or context tokens. This model extends this concept to do multiple-span extraction with BERT from a given text and context tokens.

Example Input:

Text: "Cold has numerous physiological and pathological effects on the human body, as well as on other organisms. Cold environments may promote certain psychological traits, as well as having direct effects on the ability to move. Shivering is one of the first physiological responses to cold. Even at low temperatures, the cold can massively disrupt the blood circulation. Extracellular water freezes and tissue is destroyed."

Context Tokens: "symptoms of disease"

Multi-span Extraction from Text: "Shivering", "disrupt the blood circulation", "Extracellular water freezes and tissue is destroyed"

