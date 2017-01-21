export STANFORD_HOME=/projdata9/info_fil/lixin/stanford_nlp
export CLASSPATH="$CLASS_PATH:$STANFORD_HOME/stanford_postagger/stanford-postagger.jar:$STANFORD_HOME/stanford_parser/stanford-parser.jar:$STANFORD_HOME/stanford_parser/stanford-parser-3.7.0-models.jar"
export STANFORD_MODELS="$STANFORD_HOME/stanford_postagger/models"
python preprocess.py