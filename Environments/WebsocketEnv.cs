using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;

namespace HKRL.Environments
{
	public abstract class WebsocketEnv
	{
		public Utils.Socket socket;
		protected bool _terminate = false;

		public WebsocketEnv(string url, params string[] protocols) {
			socket = new Utils.Socket(url, protocols);
		}

		protected void Connect () {
			socket.Connect();
			HKRL.Instance.Log("Connected to server");
		}

		protected abstract IEnumerator Setup();

		protected abstract IEnumerator Dispose();
		
		protected abstract IEnumerator OnMessage(Utils.Message message);

		private IEnumerator _runtime() {
			yield return Setup();
			while (true) {
				yield return new Utils.Socket.WaitForMessage(socket);
				var message = socket.UnreadMessages.Dequeue();
				yield return OnMessage(message);
				if (_terminate) break;
			}
			yield return Dispose();

		}

		protected void SendMessage(Utils.Message message) {
			message.sender = "client";
			socket.Send(message);
		}

		public void Start() {
			GameManager.instance.StartCoroutine(_runtime());
		}

		protected void CloseSocket() {
			socket.Close();
		}

		public void Close() {
			_terminate = true;
		}

	}
}