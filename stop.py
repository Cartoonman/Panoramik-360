import redis
import os


if __name__ == '__main__':
    r = redis.from_url(os.environ.get("REDIS_URL"))
    r.set('det_status', 'STOP')
    print ("done")
