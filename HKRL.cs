using Modding;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UObject = UnityEngine.Object;
using HKRL.Utils;

namespace HKRL
{
	internal class HKRL : Mod
	{
		internal static HKRL Instance { get; private set; }
        internal Environments.BasicEnv env = new("ws://localhost:8080");
		public HKRL() : base("HKRL") { }

		public override string GetVersion()
		{
			return Assembly.GetExecutingAssembly().GetName().Version.ToString();
		}

		public override void Initialize()
		{
			Log("Initializing");

			Instance = this;

			Log("Initialized");

			ModHooks.HeroUpdateHook += () =>
			{
				if (Input.GetKeyDown(KeyCode.F1))
				{
                    Log("Starting");
					env.Start();
                    
				}
				if (Input.GetKeyDown(KeyCode.F2))
				{
					Log(env.socket.IsAlive.ToString());
					Log(env.socket.UnreadMessages.Count.ToString());
				}
				if (Input.GetKeyDown(KeyCode.F3))
				{
					Log("Closing");
					env.Close();
				}
			};

		}
	}
}