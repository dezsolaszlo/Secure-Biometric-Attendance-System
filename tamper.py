import os
import sys
from store import ProfileStore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


with ProfileStore() as store:
    try:
        profiles = store.list_all()

        if not profiles:
            print("No profiles found in the database.")
        else:
            for p in profiles:
                print(f"  Name     : {p['full_name']}")
                print(f"  CNP      : {p['cnp']}")
                print(f"  Enrolled : {p['enrolled_at']}")

                emb_len = len(p['embedding']) if p['embedding'] else 0
                print(f"  Embedding: {emb_len} floats read successfully\n")

        print(f"Success: Read {len(profiles)} profiles safely. No tampering detected.")

    except Exception as error:
        print("\nTAMPER DETECTED!")
        print(f"Fernet HMAC verification failed: {error}")