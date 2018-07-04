import asyncio
import uvloop
import sqlite3
import aiohttp
import pickle
import json
import datetime
from tg_prediction import pred_agent
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/data/i.anokhin/convai/experiments/memory_nn_batch/20180703-225239/')
parser.add_argument('--data_dir', default='/data/i.fursov/convai/data')
parser.add_argument('--train_knn', default='Y')
parser.add_argument('--test_tg', default='Y')


def check_db(connection):
    return connection.execute(
        '''
            select count(1)
              from sqlite_master
             where type = ?
               and name = ?;
        ''',
        ('table', 'messages')
    ).fetchone() == (1,)


def setup_db(connection):
    connection.execute('''
        create table messages (
            chat_id bigint,
            message_id bigint,
            source text,
            text text,
            created_at timestamp
        );
    ''')


def get_context(connection, chat_id, timestamp, limit=3):
    return connection.execute(
        '''
            select text
              from messages
             where chat_id = ?
               and created_at < ?
             order by created_at desc
             limit ?;
        ''',
        (chat_id, timestamp, limit)
    ).fetchall()[::-1]


def get_facts(connection, chat_id):
    lines = connection.execute(
        '''
            select text
              from messages
             where chat_id = ?
             order by created_at
             limit 1;
        ''',
        (chat_id,)
    ).fetchone()[0].split('\n')
    if lines[0].strip().lower() != '/start':
        return []
    return lines[1:]


async def get_updates(url, retry_timeout):
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    j = await resp.json()
                    if j.get('ok') == True and len(j['result']) > 0:
                        print(j['result'])
                        return j['result']
                    await asyncio.sleep(retry_timeout)
                    continue


async def send_message(url, chat_id, text):
    j = {
        'chat_id': chat_id,
        'text': json.dumps({
            'text': text
        })
    }
    print(j)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=j) as resp:
            if resp.status == 200:
                return True


def save_message(connection, chat_id, message_id, text):
    created_at = datetime.datetime.now()
    connection.execute(
        '''
            insert into messages (
                chat_id,
                message_id,
                source,
                text,
                created_at
            ) values (?, ?, ?, ?, ?);
        ''',
        (
            chat_id,
            message_id,
            'user',
            text,
            created_at
        )
    )
    return created_at


def save_answer(connection, chat_id, text):
    created_at = datetime.datetime.now()
    connection.execute(
        '''
            insert into messages (
                chat_id,
                source,
                text,
                created_at
            ) values (?, ?, ?, ?);
        ''',
        (
            chat_id,
            'bot',
            text,
            created_at
        )
    )
    return created_at


def form_data(connection, chat_id, text, created_at):
    return {
        'context': get_context(
            connection,
            chat_id,
            created_at,
        ),
        'question': text,
        'facts': get_facts(
            connection,
            chat_id
        )
    }


async def process_updates(updates, connection, loop, send_message_url):
    answers = []
    for update in updates:
        chat_id = update['message']['chat']['id']
        message_id = update['message']['message_id']
        text = update['message']['text']
        created_at = save_message(connection, chat_id, message_id, text)
        #if not texts.startswith('/start'):
        answers.append((
            chat_id,
            loop.run_in_executor(
                None,
                get_answer,
                form_data(connection, chat_id, text, created_at)
            )
        ))
    sends = []
    for chat_id, answer in answers:
        answer_text = await answer
        sends.append(send_message(send_message_url, chat_id, answer_text))
        save_answer(connection, chat_id, answer_text)
    for send in sends:
        await send


async def main(loop, connection, get_updates_url, send_message_url):
    while True:
        asyncio.ensure_future(
            process_updates(
                await get_updates(get_updates_url, 0.5),
                connection, loop, send_message_url
            ),
            loop=loop
        )


def get_answer(data):
    print('asdasfa')
    if args.test_tg == 'N':
        answer = agent.predict(data)
        return '¯\_(ツ)_/¯' + answer
    else:
        return '¯\_(ツ)_/¯'


if __name__ == '__main__':
    args = parser.parse_args()

    if args.train_knn == 'Y':
        train_model = True
    else:
        train_model = False

    if args.test_tg == 'N':
        raw_utts = pickle.load(open(os.path.join(args.data_dir, 'raw_responses.pkl'), 'rb'))
        emb_path = os.path.join(args.model_dir, 'embeddings.pkl')
        agent = pred_agent(args, raw_utts, emb_path, train_model)

    bot_token = '9a1233af-e913-4b47-9ca9-a61851475454'  #os.environ['BOT_TOKEN']
    print('lets go!')
    connection = sqlite3.connect('loopai.db')
    if not check_db(connection):
        setup_db(connection)
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        main(
            loop,
            connection,
            f'https://2258.lnsigo.mipt.ru/bot{bot_token}/getUpdates',
            f'https://2258.lnsigo.mipt.ru/bot{bot_token}/sendMessage'
        )
    )
