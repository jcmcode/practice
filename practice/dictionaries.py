

dictionary = {"Canada": "Ottawa",
              "France": "Paris",
              "Ireland": "Dublin"}


#print(dir(dictionary)) # gets all the attributes and methods of dictionary

#print(help(dictionary)) # explains attributes and methods of the dictionary



if dictionary.get("Russia"): # returns None which isnt truthy thus else block runs
    print("exists")
else:
    print("does not exist")
    
    
    
dictionary.update({"Germany": "Berlin"}) #adds to dictionary

dictionary.update({"Canada": "Toronto"}) #updates key in dictionary

dictionary.pop("Ireland") #removes

keys = dictionary.keys()


for key in dictionary.keys():
    print(key)
    
for value in dictionary.values(): 
    print(value)
    
    
items = dictionary.items()

for key, value in dictionary.items():
    print(f"{key}: {value}")