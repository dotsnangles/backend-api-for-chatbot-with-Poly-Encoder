### Todos

1. MySQL CRUD Operations with SQLAlchemy
2. Object Serialization and Data Deserialization with Marshmallow
3. REST API with Flask and the above

### Download the files for demo

```bash
python download.py
```

### Set DB
Ubuntu 20.04.4 LTS on WSL
```bash
sudo apt install mysql-server
sudo service mysql start
sudo service mysql status
sudo mysql -u root -p
```

```mysql
mysql> CREATE USER 'user'@'localhost' IDENTIFIED BY 'password';
mysql> GRANT ALL ON *.* TO 'user'@'localhost' WITH GRANT OPTION;
mysql> CREATE DATABASE chatbot;
```

### On Colab

![Link](https://colab.research.google.com/github/dotsnangles/chatbot-rest-api/blob/master/on_colab.ipynb)