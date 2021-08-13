import json
import aicmder as cmder


def test_image():
    # curl 127.0.0.1:8099/predict -X POST -d '{"img_base64": "你是谁"}'
    print(cmder)
    config = {'image': {'name': './ImageModule', 'init_args': {'file_path': './tests_model/config.yaml'}}}
    serve = cmder.serve.ServeCommand()
    serve.execute(['-w', '1', '-c', json.dumps(config), '-p', '8099', '--max_connect', '1'])

if __name__ == '__main__':
    test_image()