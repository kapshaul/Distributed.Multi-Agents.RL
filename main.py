import configparser

from PPO.ppo import ppo



CONFIG_PATH_PPO = "PPO/config.ini"

def main():
    # Load configuration
    config = configparser.ConfigParser()

    config.read(CONFIG_PATH_PPO)
    ppo(config=config)


if __name__ == "__main__":
    main()