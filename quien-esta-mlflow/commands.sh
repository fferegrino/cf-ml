# First train attempt
python training.py dialogos.csv 1000 true 1000 balanced

# Second train attempt
python training.py dialogos.csv 500 true 1000 balanced

# Third train attempt
python training.py dialogos.csv 1500 true 1500 balanced

# ...
docker-compose --env-file default.env up
