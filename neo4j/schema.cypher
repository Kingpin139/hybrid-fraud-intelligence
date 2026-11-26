// Neo4j Schema for Hybrid Fraud Intelligence
// Relational Behavior Graph

// Constraints
CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT card_id IF NOT EXISTS FOR (c:Card) REQUIRE c.card_id IS UNIQUE;
CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE;
CREATE CONSTRAINT ip_addr IF NOT EXISTS FOR (i:IP) REQUIRE i.address IS UNIQUE;
CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE;
CREATE CONSTRAINT phone_id IF NOT EXISTS FOR (p:Phone) REQUIRE p.phone_id IS UNIQUE;
CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.email_id IS UNIQUE;
CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE;

// Indexes for temporal queries
CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp);
CREATE INDEX user_created IF NOT EXISTS FOR (u:User) ON (u.created_at);

// Apply with (docker compose mounts ./neo4j at /schema):
//   docker compose exec neo4j sh -c 'cypher-shell -u neo4j -p "${NEO4J_AUTH#neo4j/}" -f /schema/schema.cypher'
//
// Mapping to the PyG HeteroData schema used by RBGNN (src/hybrid_fraud/models/rb_gnn.py):
//   (:User)-[:TRANSACTS]->(:Merchant) -> ('user','transacts','merchant')  (aggregated from Transaction nodes)
//   (:User)-[:USES]->(:Device)        -> ('user','uses','device')
//   (:User)-[:FROM]->(:IP)            -> ('user','from','ip')
//   (:User)-[:OWNS]->(:Card)          -> ('user','owns','card')
//   (:User)-[:SHARES]->(:Phone|:Email)-> ('user','shares','phone'|'email')
//   Reverse edges ('rev_*') are added when building HeteroData.
//
// Example data model relationships:
// (:User)-[:PERFORMS]->(:Transaction)-[:TO]->(:Merchant)
// (:User)-[:USES]->(:Device)
// (:User)-[:FROM]->(:IP)
// (:User)-[:OWNS]->(:Card)
// (:User)-[:SHARES]->(:Phone)
// (:User)-[:SHARES]->(:Email)
// (:User)-[:SHARES]->(:Address)

// Sample fraud ring detection query
// MATCH path = (a:User)-[:USES|FROM|OWNS|SHARES*1..3]-(shared)
// WHERE size([(shared)<-[:USES|FROM|OWNS|SHARES]-(other:User) | other]) > 3
// RETURN path LIMIT 50;
