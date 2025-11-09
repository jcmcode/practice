

groceries = [["apple", "banana", "pear"], ["celery", "carrots", "potatoes"], ["chicken", "fish", "turkey"]]

for collection in groceries:
    print(collection)
    
    
for group in groceries:
    for food in group:
        print(food, end=" ")
    print()