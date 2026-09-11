from app.rag.embedding import MockEmbeddingProvider


def main():

    provider = MockEmbeddingProvider(
        dimensions=384
    )

    text = """
    Employees are entitled to 20 days
    of annual leave per year.
    """

    embedding = provider.embed(text)

    print("Embedding generated successfully")
    print("Dimensions:", len(embedding))
    print("First 10 values:", embedding[:10])

    print(
        "Vector magnitude:",
        sum(value * value for value in embedding) ** 0.5
    )


if __name__ == "__main__":
    main()