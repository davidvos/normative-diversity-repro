import numpy as np
from dart.external.discount import harmonic_number
from dart.external.kl_divergence import compute_kl_divergence
from collections import defaultdict


class Representation:
    """
    Calculates Representation of entities linked to different political parties using KL Divergence.
    Currently the implementation is only suitable for the Participatory model, where we compare the party distribution
    in the recommendations to the one that was available in the pool. For the Deliberative and Critical models we also
    need to implement comparison with an equal and "pool inverse" distribution
    """

    def __init__(self, config):
        self.config = config
        self.party_data = {}

        all_political_parties = {
            'us':
                np.array([
                ["Democrat", "Democratic Party"],
                ["Republican", "Republican Party"],
                ]),
            'nl':
                np.array([
                    ["VVD", "Volkspartij voor Vrijheid en Democratie"],
                    ["D66", "Democraten 66"],
                    ["PVV", "Partij voor de Vrijheid"],
                    ["CDA", "Christen-Democratisch Appèl"],
                    ["SP", "Socialistische Partij"],
                    ["PvdA", "Partij van de Arbeid"],
                    ["GL", "GroenLinks"],
                    ["PvdD", "Partij voor de Dieren"],
                    ["CU", "ChristenUnie"],
                    ["FVD", "Forum voor Democratie"],
                    ["JA21", "JA21"],
                    ["SGP", "Staatkundig Gereformeerde Partij"],
                    ["DENK", "DENK"],
                    ["Volt", "Volt Nederland"],
                    ["BBB", "BoerBurgerBeweging"],
                    ["BIJ1", "BIJ1"],
                    ["50PLUS", "50PLUS"]
                ])
        }
        self.political_parties = all_political_parties[config['country']]

    @staticmethod
    def in_entities(entities, party):
        """
        Check if the political party is mentioned in line with the article's entities.
        This means checking the political party that mentioned politicians belong to. Could be extended to also checking
        mentioned organisations, but checking the fulltext for this proved more efficient.
        """
        # TODO: discuss whether it should be a count of mentions per article or a binary value per article!
        short_party = party[0]
        full_party = party[1]
        # only consider entities of type person
        persons = filter(lambda x: x['label'] == 'PERSON', entities)
        count = 0
        for person in persons:
            # if the person has a property 'party' (this avoids checking this value for people that are not politicians)
            if 'parties' in person:
                if short_party in person['parties'] or full_party in person['parties']:
                    count += len(person['spans'])

        organisations = filter(lambda x: x['label'] == 'ORG', entities)
        for organisation in organisations:
            # if the person has a property 'party' (this avoids checking this value for people that are not politicians)
            if short_party == organisation['text'] or full_party == organisation['text']:
                count += len(organisation['spans'])
        return count

    def compute_distr(self, articles, adjusted=False):
        """Compute the genre distribution for a given list of Items."""
        n = len(articles)
        sum_one_over_ranks = harmonic_number(n)
        rank = 0
        distr = {}
       
        for indx, entities in enumerate([x['enriched_entities'] for x in articles]):
            total = 0
            rank += 1
            d = defaultdict(int)
            # for each party specified in the configuration file
            if entities and len(entities) > 0:

                persons = [x for x in entities if x['label'] == 'PERSON']
                
                for person in persons:
                    if 'party' in person and person['party']:
                        d[person['party'][0]] += len(person['spans'])
                        total += len(person['spans'])

            for party, mentions in d.items():
                    party_freq = distr.get(party, 0.)
                    distr[party] = party_freq + mentions / total * 1 / rank / sum_one_over_ranks if adjusted else party_freq + mentions / total

        if sum(distr.values()) > 0:
            factor = 1.0 / sum(distr.values())
            for key, value in distr.items():
                distr[key] = value * factor

        return distr

    def calculate(self, pool, recommendation):
        pool_news = [x for x in pool if x['category'] == 'news']
        recommendation_news = [x for x in recommendation if x['category'] == 'news']

        if len(pool_news) > 0 and len(recommendation_news) > 0:
            pool_vector = self.compute_distr(pool_news, adjusted=False)
            recommendation_vector = self.compute_distr(recommendation_news, adjusted=True)
            if pool_vector and recommendation_vector:
                divergence_with_discount =  compute_kl_divergence(pool_vector, recommendation_vector)

                recommendation_vector = self.compute_distr(recommendation_news, adjusted=False)
                divergence_without_discount = compute_kl_divergence(pool_vector, recommendation_vector)
                return [divergence_with_discount, divergence_without_discount]
            else:
                return
        else:
            return
