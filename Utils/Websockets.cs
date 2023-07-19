using WebSocketSharp;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;

namespace HKRL.Utils
{
	/// <summary>
	/// A wrapper around WebSocketSharp.WebSocket that adds a few additional utilities
	/// </summary>
	public class Socket : WebSocket
	{
		public Queue<Message> UnreadMessages { get; private set; } = new Queue<Message>();
		public Message LastMessageSent { get; private set; }
		public Socket(string url, params string[] protocols) : base(url, protocols)
		{
			this.OnMessage += (sender, e) =>
			{
				Message m = JsonConvert.DeserializeObject<Message>(e.Data);
				// HKRL.Instance.Log("Received message: " + m.type);
				UnreadMessages.Enqueue(m);

			};
		}

		public void Send(Message data)
		{
			data.sender = "client";
			string textData = JsonConvert.SerializeObject(data);
			base.Send(textData);
			LastMessageSent = data;
		}

		public void RawSend(string data)
		{
			base.Send(data);
		}

		public void SendAsync(Message data, System.Action<bool> completed)
		{
			data.sender = "client";
			string textData = JsonConvert.SerializeObject(data);
			base.SendAsync(textData, completed);
			LastMessageSent = data;
		}

		public class WaitForMessage : CustomYieldInstruction
		{
			private Socket socket;
			private bool wait = true;

			public WaitForMessage(Socket socket)
			{
				this.socket = socket;
			}
			public override bool keepWaiting
			{
				get
				{
					return socket.UnreadMessages.Count == 0;
				}
			}
		}
	}

	public class Message
	{
		public string type;
		public string sender;
		public MessageData data;

	}

	public struct MessageData
	{
		public int[] state_size;
		public int? action_size;
		public string level;
		public int? frames_per_wait;
		public int[] state;
		public int? reward;
		public string info;
		public bool? done;
		public int? action_taken;
		public int[] prev_state;
		public int? action;
		public int? knight_health;
		public int? enemy_health;
		public int? soul_level;
		public int? time_scale;
		public bool? frame_stack;
	}
}