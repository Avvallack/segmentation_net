import os
import boto3

from pyathena import connect
from dotenv import load_dotenv


load_dotenv()
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

boto_sess = boto3.Session(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key
)
my_region = boto_sess.region_name

conn = connect(
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_access_key,
    s3_staging_dir='s3://audigent-fingerprinting-s3-endpoint-dev/',
    region_name=my_region
)


query = """
select
distinct
e.partner_site_host,
e.partner_site_path,
e.partner_site_query,
e.visit_user_agent,
e.visit_ip_address,
e.visit_referrer,
e.country,
e.state,
e.city,
e.postcode,
e.latitude,
e.longitude,
e.browser_type,
e.device_type,
e.halo_id,
e.segment_id

from "audigent_data_v2_partners_prod"."epam_visits_with_halo_id_export" as e
join
(
WITH halo_count (halo_id,  partner_site_host, segments_per_visit) 
AS
(
SELECT halo_id,  partner_site_host, count(distinct segment_id) as segments_per_visit
FROM "audigent_data_v2_partners_prod"."epam_visits_with_halo_id_export"
WHERE audigent_id != '9bc48c36-1f73-45b4-ad8c-e714831836ac'
GROUP BY halo_id, partner_site_host
)
SELECT distinct halo_id, partner_site_host from halo_count
where segments_per_visit = 1
) as t
on e.halo_id = t.halo_id and e.partner_site_host=t.partner_site_host
"""