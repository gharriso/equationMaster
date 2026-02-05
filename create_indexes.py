#!/usr/bin/env python3
"""
Create MongoDB indexes for efficient querying.
"""

from pymongo import MongoClient, ASCENDING

from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME


def main():
    print(f"Connecting to MongoDB at {MONGODB_URI}...")
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    print("Creating indexes...")

    # Index on chapter for filtering
    collection.create_index([("chapter", ASCENDING)], name="chapter_idx")
    print("  Created index: chapter_idx")

    # Index on reference for sorting
    collection.create_index([("reference", ASCENDING)], name="reference_idx")
    print("  Created index: reference_idx")

    # Index on name for searching
    collection.create_index([("name", ASCENDING)], name="name_idx")
    print("  Created index: name_idx")

    # Index on variables.symbol for finding equations by variable
    collection.create_index([("variables.symbol", ASCENDING)], name="variable_symbol_idx")
    print("  Created index: variable_symbol_idx")

    # Compound index for chapter + reference sorting
    collection.create_index(
        [("chapter", ASCENDING), ("reference", ASCENDING)],
        name="chapter_reference_idx"
    )
    print("  Created index: chapter_reference_idx")

    # List all indexes
    print("\nCurrent indexes:")
    for index in collection.list_indexes():
        print(f"  {index['name']}: {index['key']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
