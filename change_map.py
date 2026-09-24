import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Change the map
client.load_world("Town05")

print("Map changed to Town05")
