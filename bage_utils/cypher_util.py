import json


class CypherUtil(object):
    """ Neo4j Cypher """

    @staticmethod
    def escape(query):
        return json.dumps(query)
