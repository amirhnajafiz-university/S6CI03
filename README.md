# Neuroevolution Game

Snail jumper game with Neural Networks and Generic Algorithm.

<p align="center">
    <img src="SnailJumper.png" width="700" />
</p>

In this project we are going to play the Jumping Snail game with 
our Neural Network that does not use backpropagation; insted it uses 
generic algorithm to find the best playing algorithm.

When you want to use the NN mode of the game, it creates 300 agents and 
starts our Neural Network training. After many generations, it finaly gets
to a logical algorithm, but it is not the best algorithm.

## How to use?
Clone the project:
```shell
git clone https://github.com/amirhnajafiz/neuroevolution-game.git
```

Install requirements:
```shell
pip install -r requirements.txt
```

Now you can run the game:
```shell
python game.py
```

If you want you can check the generations by the following command:
```shell
python history.py
```
